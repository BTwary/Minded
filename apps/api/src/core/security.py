"""Authentication and Security Utilities with Strict JWT Token Verification & Tenancy Enforcement."""
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Union
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import hashlib

# Compatibility bridge: bcrypt >= 4.0 removed __about__ and rejects passwords > 72 bytes.
# passlib 1.7.4 probes wrap bugs with 255-byte secrets unless backend workarounds are initialized.
try:
    import bcrypt
    if not hasattr(bcrypt, "__about__"):
        class _BcryptAbout:
            __version__ = getattr(bcrypt, "__version__", "4.0.1")
        bcrypt.__about__ = _BcryptAbout()
    import passlib.handlers.bcrypt as _phb
    _phb._BcryptBackend._workrounds_initialized = True
except Exception:
    pass

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from apps.api.src.core.config import settings
from apps.api.src.core.database import get_db

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"
security_bearer = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify bcrypt hashes; accept legacy SHA-256 hashes only for migration."""
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
        return _pwd_context.verify(plain_password, hashed_password)
    return _legacy_password_hash(plain_password) == hashed_password


def get_password_hash(password: str) -> str:
    """Generate a production-suitable bcrypt password hash."""
    return _pwd_context.hash(password)


def _legacy_password_hash(password: str) -> str:
    """Legacy deterministic hash used only to verify pre-hardening local accounts."""
    salt = settings.SECRET_KEY[:32]
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def needs_password_rehash(hashed_password: str) -> bool:
    return not hashed_password.startswith(("$2a$", "$2b$", "$2y$")) or _pwd_context.needs_update(hashed_password)


def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate signed JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[str]:
    """Decode JWT and return user ID subject."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        return user_id
    except JWTError:
        return None


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
):
    """Strictly enforce JWT authentication and return authenticated User entity. Zero demo bypass."""
    from apps.api.src.models.entities import User

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided or are invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = decode_access_token(auth.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user account not found.",
        )
    if not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive or has been deactivated.",
        )

    return user


def get_optional_user(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
):
    """Optional authentication for public/preview endpoints."""
    from apps.api.src.models.entities import User
    if not auth or not auth.credentials:
        return None
    user_id = decode_access_token(auth.credentials)
    if not user_id:
        return None
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not getattr(user, "is_active", True):
        return None
    return user


def get_user_authorized_project_ids(current_user, db: Session) -> List[str]:
    """Return all project IDs accessible to the authenticated user to prevent cross-tenant enumeration."""
    from apps.api.src.models.entities import Project
    from apps.api.src.core.config import settings

    # Role separation:
    # 1. 'superadmin' has platform-wide authority across all projects.
    # 2. 'organization_admin' or 'admin' is scoped to projects within their organization.
    # 3. 'analyst' (default) is scoped strictly to projects they own.
    if current_user.role == "superadmin":
        return [p.id for p in db.query(Project.id).all()]

    if current_user.role in ("organization_admin", "admin"):
        org_id = getattr(current_user, "organization_id", None)
        if org_id:
            projects = (
                db.query(Project.id)
                .filter((Project.org_id == org_id) | (Project.owner_id == current_user.id))
                .all()
            )
        else:
            # Fail closed: missing organization membership denies org-level access,
            # scoping strictly to projects personally owned by the user.
            projects = (
                db.query(Project.id)
                .filter(Project.owner_id == current_user.id)
                .all()
            )
        return [p[0] for p in projects]

    if settings.DEMO_MODE:
        projects = (
            db.query(Project.id)
            .filter((Project.owner_id == current_user.id) | (Project.id == "proj-default"))
            .all()
        )
    else:
        projects = (
            db.query(Project.id)
            .filter(Project.owner_id == current_user.id)
            .all()
        )
    return [p[0] for p in projects]


def verify_project_ownership(project_id: str, current_user, db: Session):
    """Enforce multi-tenant project ownership boundary."""
    from apps.api.src.models.entities import Project
    from apps.api.src.core.config import settings

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    is_demo = settings.DEMO_MODE and project.id == "proj-default"
    if project.owner_id and project.owner_id != current_user.id and not is_demo:
        if current_user.role == "superadmin":
            pass
        elif current_user.role in ("organization_admin", "admin"):
            org_id = getattr(current_user, "organization_id", None)
            if not org_id or project.org_id != org_id:
                raise HTTPException(status_code=403, detail="Unauthorized: You do not have access to this project.")
        else:
            raise HTTPException(status_code=403, detail="Unauthorized: You do not have access to this project.")
    return project


def verify_dataset_ownership(dataset_id: str, current_user, db: Session):
    """Enforce multi-tenant dataset ownership boundary."""
    from apps.api.src.models.entities import Dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if dataset.project_id:
        verify_project_ownership(dataset.project_id, current_user, db)
    return dataset
