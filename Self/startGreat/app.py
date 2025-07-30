from fastapi import FastAPI
from data_harvester.data_harvester import router as hello_router  # import your router

app = FastAPI(
    title="Multi-file FastAPI",
    version="0.1.0",
    description="Example of splitting routes into modules",
)

# include the router defined in routers/hello.py
app.include_router(hello_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
