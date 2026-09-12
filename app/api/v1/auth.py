"""
Human sign-in - signup/login backed by app.core.auth_db's dedicated
Postgres database, issuing JWT access tokens on successful login. This
is the OTHER half of app/core/auth.py's unified principal system - the
human-facing side, as opposed to the long-lived service API keys
scripts/create_api_key.py issues.
"""
from pydantic import BaseModel, EmailStr, field_validator
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone

from app.core.auth_db import get_auth_db, User
from app.core.auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

VALID_SIGNUP_ROLES = {"cs_agent", "readonly"}
# Deliberately excludes "admin" and "service" from self-service signup:
# admin is the highest-privilege role in this system (data-destructive
# tooling, threshold changes, policy uploads) and shouldn't be
# grantable by simply filling out a form: a real deployment promotes a
# user to admin through a separate, deliberate action (another admin
# doing it, or a database migration at initial setup) - not a public
# signup endpoint. "service" isn't a human role at all (see
# scripts/create_api_key.py for how service credentials are actually
# issued).


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "cs_agent"

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("role")
    @classmethod
    def role_must_be_self_serviceable(cls, v):
        if v not in VALID_SIGNUP_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_SIGNUP_ROLES)} - admin/service accounts are not self-service")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_auth_db)):
    user = User(
        email=payload.email, hashed_password=hash_password(payload.password),
        name=payload.name, role=payload.role,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    return TokenResponse(access_token=token, role=user.role, name=user.name)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_auth_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    # Deliberately the SAME error message whether the email doesn't
    # exist or the password is wrong - distinguishing them lets an
    # attacker enumerate valid email addresses, a real, well-known
    # authentication anti-pattern this avoids on purpose.
    if user is None or not user.active or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_access_token(user_id=user.id, email=user.email, role=user.role)
    return TokenResponse(access_token=token, role=user.role, name=user.name)
