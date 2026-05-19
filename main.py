from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

app = FastAPI(title="CrewAI Research Agent")

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
            goal="Fetch reliable web information about the topic",
            backstory="You are an expert web researcher.",
            tools=[search_tool],
            llm=llm,
            verbose=True,
        )

        summarizer = Agent(
            role="Summarizing Agent",
            goal="Summarize fetched information clearly",
            backstory="You convert research into simple key points.",
            llm=llm,
            verbose=True,
        )

        report_writer = Agent(
            role="Report Writing Agent",
            goal="Write a professional research report",
            backstory="You write structured reports with headings and conclusion.",
            llm=llm,
            verbose=True,
        )

        reviewer = Agent(
            role="Review Agent",
            goal="Review and improve the final report",
            backstory="You improve grammar, clarity, accuracy, and structure.",
            llm=llm,
            verbose=True,
        )

        fetch_task = Task(
            description="Fetch reliable information about: {topic}",
            expected_output="Research notes with facts, examples, and useful sources.",
            agent=fetcher,
        )

        summary_task = Task(
            description="Summarize the fetched research into clear bullet points.",
            expected_output="Clear summary of the most important points.",
            agent=summarizer,
        )

        report_task = Task(
            description="Write a full research report using the summary.",
            expected_output="Structured report with title, introduction, main points, examples, and conclusion.",
            agent=report_writer,
        )

        review_task = Task(
            description="Review and polish the report.",
            expected_output="Final polished research report.",
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