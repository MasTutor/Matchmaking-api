from fastapi import FastAPI, HTTPException
from function import *

# Import your main_workflow function from your existing script or module

app = FastAPI()

@app.get("/matchmaking/")
def run_matchmaking(email: str, category: str):
    try:
        results = main_workflow(email, category)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
