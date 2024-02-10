from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from joblib import load 
from Movies import search

app = FastAPI()
vectorizer=load('vect.joblib')
# Mount the directory containing static files (e.g., HTML templates)
app.mount("/template", StaticFiles(directory="template"), name="template")
templates = Jinja2Templates(directory="template")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    results = search("speed", vectorizer)
    print(results.drop("status",axis=1))
    return templates.TemplateResponse("int1.html", {"request": request,"results": results.drop("status",axis=1)})

@app.post("/process_input")
async def process_input( request: Request,q: str = Form(...)):  # Include 'request' parameter
    results = search(q, vectorizer)  # Get search results
    print(results)  # Print search results
    return templates.TemplateResponse("int1.html", {"request": request, "results": results.drop("status",axis=1)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
#print("speed")    
#print(search("speed",vectorizer).columns)
