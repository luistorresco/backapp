from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import matrix

app = FastAPI(title="Matrix Solver API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matrix.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Matrix Solver API"}
