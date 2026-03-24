import os
import json
from typing import Optional

from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types

# Load values from .env into environment variables
load_dotenv()

st.set_page_config(page_title="HR Agent Playground", page_icon="🤖", layout="wide")


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Add it to your .env file."
        )
    return genai.Client(api_key=api_key)


HR_TEMPLATES = {
    "Talent Acquisition Agent": """You are an HR talent acquisition agent.
Your job is to help recruiters design job descriptions, screening criteria,
interview questions, and candidate evaluation summaries.
Be practical, concise, and structured.
Always note when human review is required.
Do not make employment decisions; support decision-making only.""",

    "HR Policy Assistant": """You are an HR policy assistant.
Your job is to explain HR policy language, summarize policy documents,
draft employee-facing answers, and flag unclear or risky wording.
Be neutral, clear, and careful.
If legal review may be needed, say so explicitly.""",

    "Learning and Development Coach": """You are an HR learning and development agent.
Your job is to create capability-building plans, manager coaching tips,
training outlines, and role-based learning pathways.
Be practical and business-oriented.
Tailor outputs to the audience and skill level.""",

    "Employee Support Triage Agent": """You are an employee support triage agent.
Your job is to classify incoming HR questions, suggest next actions,
draft empathetic responses, and route the issue to the right team.
Do not pretend to be a human caseworker.
Be empathetic, structured, and privacy-conscious.""",
}


def build_system_prompt(
    base_instructions: str,
    tone: str,
    output_format: str,
    use_guardrails: bool,
) -> str:
    guardrails = """
Additional guardrails:
- Do not fabricate company policies or legal rules.
- Highlight assumptions clearly.
- Flag sensitive cases involving discrimination, harassment, termination, compensation, health, or legal escalation.
- Keep personally identifiable information to a minimum.
- Recommend human review where appropriate.
""" if use_guardrails else ""

    return f"""
{base_instructions}

Response tone: {tone}
Preferred output format: {output_format}
{guardrails}
""".strip()


def call_model(
    model_name: str,
    system_prompt: str,
    user_task: str,
    context_blob: Optional[str] = None,
) -> str:
    client = get_client()

    contents = []
    if context_blob:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Context for this task:\n\n{context_blob}")],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_task)],
        )
    )

    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
        ),
        contents=contents,
    )

    return response.text or "No response returned."


st.title("🤖 HR Agent Playground")
st.caption("A lightweight workshop demo for testing HR-focused AI agents with a Google model API.")

with st.sidebar:
    st.header("Agent Setup")
    template = st.selectbox("Choose an agent template", list(HR_TEMPLATES.keys()))
    model_name = st.text_input("Model name", value="gemini-2.5-pro")
    tone = st.selectbox("Tone", ["Professional", "Empathetic", "Executive", "Concise"], index=0)
    output_format = st.selectbox(
        "Output format",
        ["Bullets", "Table", "Memo", "Email draft", "Decision tree"],
        index=0,
    )
    use_guardrails = st.checkbox("Enable HR guardrails", value=True)

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Define the agent")
    base_instructions = st.text_area(
        "Agent instructions",
        value=HR_TEMPLATES[template],
        height=240,
    )

    st.subheader("2) Add scenario context")
    context_blob = st.text_area(
        "Optional context",
        placeholder="Paste a policy snippet, job description, employee question, or workshop scenario here.",
        height=180,
    )

with right:
    st.subheader("3) Give the agent a task")
    user_task = st.text_area(
        "Task",
        value="Draft a first-pass screening rubric for a People Analytics Manager role, including must-have criteria, interview questions, and risks to watch for.",
        height=180,
    )

    system_prompt = build_system_prompt(base_instructions, tone, output_format, use_guardrails)

    with st.expander("Preview final system prompt"):
        st.code(system_prompt)

    run = st.button("Run agent", type="primary", use_container_width=True)

if run:
    if not user_task.strip():
        st.error("Please enter a task.")
    else:
        with st.spinner("Running agent..."):
            try:
                result = call_model(model_name, system_prompt, user_task, context_blob)

                st.subheader("4) Agent output")
                st.write(result)

                with st.expander("Save test case"):
                    payload = {
                        "template": template,
                        "model_name": model_name,
                        "system_prompt": system_prompt,
                        "context": context_blob,
                        "task": user_task,
                        "result": result,
                    }
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(payload, indent=2),
                        file_name="hr_agent_test_case.json",
                        mime="application/json",
                    )

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "**Workshop use:** participants edit the instructions, paste an HR scenario, run the agent, and compare outputs."
)