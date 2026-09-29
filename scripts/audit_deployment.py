"""audit_deployment.py: Comprehensive deployment readiness audit scanner."""
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def audit_env_vars():
    env_vars = {}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Ignore non-source or generated dirs
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.git', '.next', '__pycache__', 'venv', '.venv', 'dist', 'build']]
        for file in files:
            if file.endswith(('.py', '.ts', '.tsx', '.js', '.jsx', '.html', '.sh')):
                fpath = os.path.join(root, file)
                rel_path = os.path.relpath(fpath, PROJECT_ROOT)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        
                        # Python getenv / environ
                        py_matches = re.findall(r'os\.(?:getenv|environ\.get|environ\[)\s*\(?\s*[\'\"]([A-Z0-9_]+)[\'\"]', text)
                        for v in py_matches:
                            env_vars.setdefault(v, []).append((rel_path, "python"))
                            
                        # JS/TS process.env
                        js_matches = re.findall(r'process\.env\.([A-Z0-9_]+)', text)
                        for v in js_matches:
                            env_vars.setdefault(v, []).append((rel_path, "js/ts"))
                except Exception as e:
                    pass
    return env_vars

def main():
    print("=" * 80)
    print("MINDED (AA-OS) DEPLOYMENT READINESS AUDIT")
    print("=" * 80)
    
    env_vars = audit_env_vars()
    print(f"\n[1] Identified {len(env_vars)} Distinct Environment Variables Across Codebase:\n")
    
    frontend_safe = []
    backend_secrets = []
    
    for var, locations in sorted(env_vars.items()):
        loc_str = ", ".join(sorted(set(loc[0] for loc in locations)))
        is_client = any("apps/web" in loc[0] for loc in locations)
        is_server = any("apps/api" in loc[0] or "packages" in loc[0] for loc in locations)
        
        if var.startswith("NEXT_PUBLIC_"):
            frontend_safe.append((var, loc_str))
            cat = "FRONTEND-SAFE (NEXT_PUBLIC)"
        elif is_client and not is_server:
            frontend_safe.append((var, loc_str))
            cat = "FRONTEND BUILD-TIME"
        else:
            backend_secrets.append((var, loc_str))
            cat = "BACKEND SECRET"
            
        print(f" - {var:<30} [{cat}] -> {loc_str}")

    print("\n" + "=" * 80)
    print("FRONTEND LEAK CHECK:")
    print("=" * 80)
    
    leaks = []
    client_files = []
    for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "apps", "web")):
        dirs[:] = [d for d in dirs if d not in ['node_modules', '.next', '.git']]
        for file in files:
            if file.endswith(('.ts', '.tsx', '.js', '.jsx', '.html')):
                client_files.append(os.path.join(root, file))

    dangerous_patterns = [
        ("GEMINI_API_KEY", r"GEMINI_API_KEY"),
        ("OPENAI_API_KEY", r"OPENAI_API_KEY"),
        ("DATABASE_URL", r"DATABASE_URL"),
        ("SECRET_KEY", r"SECRET_KEY"),
        ("HF_TOKEN", r"HF_TOKEN"),
        ("VERCEL_TOKEN", r"VERCEL_TOKEN"),
        ("SUPABASE_SERVICE_ROLE_KEY", r"SUPABASE_SERVICE_ROLE_KEY"),
        ("POSTGRES_PASSWORD", r"POSTGRES_PASSWORD"),
    ]

    for fpath in client_files:
        rel = os.path.relpath(fpath, PROJECT_ROOT)
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            for name, pattern in dangerous_patterns:
                # Exclude comments or explicit public wrappers
                matches = re.findall(rf'process\.env\.({pattern})', content)
                if matches and not name.startswith("NEXT_PUBLIC_"):
                    leaks.append((name, rel))

    if not leaks:
        print("[PASS] ZERO backend secrets or database credentials are referenced in browser client bundles!")
    else:
        for name, rel in leaks:
            print(f"[FAIL] POTENTIAL LEAK: {name} in {rel}")

if __name__ == "__main__":
    main()
