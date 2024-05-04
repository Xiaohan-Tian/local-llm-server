from fastapi import APIRouter

router = APIRouter()

@router.get("/hi")
def hi():
    return {"hi": "server is running."}
