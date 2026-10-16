"""Authentication API Endpoints with Real JWT Validation."""
from datetime import datetime
import sqlite3
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from apps.api.src.core.database import get_db
from apps.api.src.core.config import settings
from apps.api.src.core.security import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
    needs_password_rehash,
)
from apps.api.src.models.entities import User

router = APIRouter(prefix="/auth", tags=["auth"])


class UserRegisterSchema(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    # Public registration may never self-assign a privileged role.
    role: str = "analyst"


class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str


class TokenSchema(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: str
    role: str


class UserProfileSchema(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    organization_id: Optional[str] = None
    created_at: Optional[datetime] = None


def _is_duplicate_email_error(error: IntegrityError) -> bool:
    """Recognize the email uniqueness violation on SQLite and PostgreSQL.

    Other unique, foreign-key, and not-null violations are persistence failures,
    not proof that the requested email already belongs to an account.
    """
    original = error.orig
    if isinstance(original, sqlite3.IntegrityError):
        return str(original) == "UNIQUE constraint failed: users.email"
    return (
        (getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)) == "23505"
        and getattr(getattr(original, "diag", None), "constraint_name", None) == "ix_users_email"
    )


@router.post("/register", response_model=TokenSchema)
def register_user(payload: UserRegisterSchema, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists.")

        user = User(
            email=payload.email,
            hashed_password=get_password_hash(payload.password),
            full_name=payload.full_name,
            role="analyst",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    except IntegrityError as error:
        db.rollback()
        if _is_duplicate_email_error(error):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists.") from None
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to register user.") from None
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to register user.") from None

    # Only the new, successfully persisted user may receive a registration token.
    token = create_access_token(subject=user.id)
    return TokenSchema(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


@router.post("/login", response_model=TokenSchema)
def login_user(payload: UserLoginSchema, db: Session = Depends(get_db)):
    """Authenticate user and return JWT."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if needs_password_rehash(user.hashed_password):
        user.hashed_password = get_password_hash(payload.password)
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(subject=user.id)
    return TokenSchema(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


@router.post("/demo-token", response_model=TokenSchema)
def get_demo_token(request: Request, db: Session = Depends(get_db)):
    """Auto-provision demo analyst user only from an explicitly local development request."""
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    client_host = request.client.host if request.client else ""
    if settings.ENVIRONMENT not in {"development", "dev", "local"} or client_host not in local_hosts:
        raise HTTPException(status_code=404, detail="Demo authentication is local-development only.")
    user = db.query(User).filter(User.id == "default-user").first()
    if not user:
        user = User(
            id="default-user",
            email="analyst@aa-os.ai",
            hashed_password=get_password_hash("demo1234"),
            full_name="Lead Data Analyst",
            role="admin",
            organization_id="default-org",
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(subject=user.id)
    return TokenSchema(
        access_token=token,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
    )


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """Return decoded authenticated user profile."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "organization_id": current_user.organization_id,
        "created_at": current_user.created_at,
    }
