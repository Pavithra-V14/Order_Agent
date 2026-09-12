"""
Frontend page routes - architecture doc 8.3's 8 pages, server-rendered
via Jinja2 (no build step / Node toolchain needed). Each page fetches its
own data client-side via the JSON API, per app/static/app.js.
"""
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"page_id": "dashboard"})


@router.get("/cases/{case_id}")
def page_case_detail(request: Request, case_id: str):
    return templates.TemplateResponse(request, "case_detail.html", {"page_id": "case_detail", "case_id": case_id})


@router.get("/escalations")
def page_escalations(request: Request):
    return templates.TemplateResponse(request, "escalations.html", {"page_id": "escalations"})


@router.get("/policies")
def page_policies(request: Request):
    return templates.TemplateResponse(request, "policies.html", {"page_id": "policies"})


@router.get("/metrics")
def page_metrics(request: Request):
    return templates.TemplateResponse(request, "metrics.html", {"page_id": "metrics"})


@router.get("/threshold-config")
def page_threshold(request: Request):
    return templates.TemplateResponse(request, "threshold.html", {"page_id": "threshold"})


@router.get("/testing")
def page_testing(request: Request):
    return templates.TemplateResponse(request, "testing.html", {"page_id": "testing"})


@router.get("/audit-log")
def page_audit(request: Request):
    return templates.TemplateResponse(request, "audit.html", {"page_id": "audit"})


@router.get("/traces/{case_id}")
def page_trace(request: Request, case_id: str):
    return templates.TemplateResponse(request, "trace.html", {"page_id": "trace", "case_id": case_id})


@router.get("/admin")
def page_admin(request: Request):
    return templates.TemplateResponse(request, "admin.html", {"page_id": "admin"})


@router.get("/login")
def page_login(request: Request):
    return templates.TemplateResponse(request, "login.html", {"page_id": "login"})


@router.get("/signup")
def page_signup(request: Request):
    return templates.TemplateResponse(request, "signup.html", {"page_id": "signup"})
