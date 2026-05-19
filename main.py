
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
    model="openrouter/openai/gpt-4o-mini",
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
            goal="Fetch reliable web information about the topic",
            backstory="You are a fast web researcher. Collect only useful facts.",
            tools=[search_tool],
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        summarizer = Agent(
            role="Summarizing Agent",
            goal="Summarize fetched information clearly",
            backstory="You convert research into short and simple key points.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        report_writer = Agent(
            role="Report Writing Agent",
            goal="Write a clean structured research report",
            backstory="You write professional reports with headings and conclusion.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        reviewer = Agent(
            role="Review Agent",
            goal="Review and polish the final report",
            backstory="You improve clarity, grammar, and structure without making it too long.",
            llm=llm,
            verbose=True,
            max_iter=1,
            allow_delegation=False,
        )

        fetch_task = Task(
            description="Fetch reliable information about: {topic}. Keep it concise.",
            expected_output="5 key facts, 3 examples, and 3 useful source names.",
            agent=fetcher,
        )

        summary_task = Task(
            description="Summarize the fetched research into simple bullet points.",
            expected_output="A short summary with 6 clear bullet points.",
            agent=summarizer,
        )

        report_task = Task(
            description="Write a clear research report from the summary.",
            expected_output="A report under 700 words with title, introduction, main points, examples, and conclusion.",
            agent=report_writer,
        )

        review_task = Task(
            description="Review and polish the report without making it longer.",
            expected_output="Final polished report under 700 words.",
            agent=reviewer,
        )

        crew = Crew(
            agents=[fetcher, summarizer, report_writer, reviewer],
            tasks=[fetch_task, summary_task, report_task, review_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff(inputs={"topic": request.topic})
        return {"topic": request.topic, "report": str(result)}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}