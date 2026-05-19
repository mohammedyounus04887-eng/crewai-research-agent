
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

app = FastAPI(title="CrewAI Research Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLM(
    model="openrouter/openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

search_tool = SerperDevTool()


class ResearchRequest(BaseModel):
    topic: str


@app.get("/")
def home():
    return {"message": "CrewAI Research Agent is running"}


@app.post("/research")
def research(request: ResearchRequest):
    try:
        fetcher = Agent(
            role="Data Fetching Agent",
            goal="Fetch useful and reliable information quickly",
            backstory="You are a fast web researcher. Collect only the most important facts.",
            tools=[search_tool],
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        summarizer = Agent(
            role="Summarizing Agent",
            goal="Summarize research into short clear points",
            backstory="You turn research notes into simple bullet points.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        report_writer = Agent(
            role="Report Writing Agent",
            goal="Write a short structured report",
            backstory="You create clean reports with title, bullet points, and conclusion.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        reviewer = Agent(
            role="Review Agent",
            goal="Polish the report without increasing length",
            backstory="You improve grammar, clarity, and flow while keeping the report short.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        fetch_task = Task(
            description="Fetch reliable information about: {topic}. Keep it very concise.",
            expected_output="Only 5 key facts and 2 source names.",
            agent=fetcher,
        )

        summary_task = Task(
            description="Summarize the fetched research into short bullet points.",
            expected_output="5 short bullet points.",
            agent=summarizer,
        )

        report_task = Task(
            description="Write a short research report from the summary.",
            expected_output="A report under 400 words with title, 5 bullet points, and conclusion.",
            agent=report_writer,
        )

        review_task = Task(
            description="Review and polish the report without increasing length.",
            expected_output="Final polished report under 400 words.",
            agent=reviewer,
        )

        crew = Crew(
            agents=[fetcher, summarizer, report_writer, reviewer],
            tasks=[fetch_task, summary_task, report_task, review_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs={"topic": request.topic})
        return {
            "topic": request.topic,
            "report": str(result)
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }