import logging
import math
import uuid
from collections import deque

from fastapi import FastAPI
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.orm import Session
from pathlib import Path

import src
from src.web.models import Base
from src.web.models import Prediction
from src.web.models import Sample
from src.model import BertTransformer
from src.web.sessionmanager import SessionManager

logfmt = "[%(asctime)s] {%(lineno)d} %(levelname)s - %(message)s"
logging.basicConfig(format=logfmt)
logger = logging.getLogger(__name__)


app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)


def load_engine():
    conn_string = f"sqlite:///{src.PATH / 'predictions.sqlite'}"
    engine = create_engine(conn_string)
    Base.metadata.create_all(engine)
    return engine


engine = load_engine()

templates = Jinja2Templates(directory="templates")
session_manager = SessionManager()
model = BertTransformer(name="luerhard/PopBERT")


def assert_user(user):
    if user:
        return user
    return uuid.uuid4().hex


def get_chain(user) -> deque:
    global session_manager
    return session_manager.get(user, deque(maxlen=100))


def translate_result(prediction: Prediction) -> list[tuple[str, float]]:
    dimensions = [
        "pop_antielite",
        "pop_pplcentr",
        # "souv_eliteless",
        # "souv_pplmore",
        # "ideol_left",
        # "ideol_right",
    ]

    results = []
    for dim in dimensions:
        val = getattr(prediction, dim)
        key = math.floor(val * 10)
        key = 9 if key == 10 else key
        results.append((f"perc{key}", round(val, 2)))

    return results


def predict(session, model, sample: Sample) -> Prediction:
    prediction = (
        session.query(Prediction)
        .filter(Prediction.sample_id == sample.id)
        .one_or_none()
    )

    if not prediction:
        result = model.predict(sample.text)[0]

        logger.warn("New Prediction for Sample %s -- %s", sample.id, result)

        prediction = Prediction(
            sample_id=sample.id,
            pop_antielite=result[0],
            pop_pplcentr=result[1],
        )
        sample.predictions.append(prediction)
        session.add(sample)
        session.commit()
    else:
        logger.warn("Cached Prediction: %s for sample: %s", prediction.id, sample.id)

    return prediction


def get_sample(session, text: str) -> Sample:
    text = text.strip()

    sample = session.query(Sample).filter(Sample.text == text).one_or_none()
    if not sample:
        sample = Sample(text=text)
        session.add(sample)

    return sample


@app.get("/high")
async def high(request: Request, sortby: str, n: int = 20, user: str | None = None):
    global engine
    global model

    logger.warn("Sorting by: %s", sortby)
    logger.warning("User: %s", user)
    user = assert_user(user)
    chain = get_chain(user)

    sorter = getattr(Prediction, sortby)

    with Session(engine) as session:
        samples = (
            session.query(Sample, Prediction)
            .join(Prediction, Sample.id == Prediction.sample_id)
            .order_by(sorter.desc())
            .limit(n)
        )
        chain.clear()
        for sample, pred in samples:
            result = translate_result(pred)
            chain.append({"message": sample.text, "result": result})

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user},
    )


@app.get("/random")
async def random(request: Request, n: int = 20, user: str | None = None):
    global engine
    global model
    global model_id

    logger.warning("User: %s", user)
    user = assert_user(user)
    chain = get_chain(user)

    chain.clear()
    with Session(engine) as session:
        samples = session.query(Sample).order_by(func.random()).limit(n)
        for sample in samples:
            pred = predict(session, model, sample)
            result = translate_result(pred)
            chain.appendleft({"message": sample.text, "result": result})

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user},
    )


@app.get("/")
async def root(request: Request, text: str = "", user: str | None = None):
    global engine
    global model

    logger.warning("User: %s", user)
    logger.warning("received text input: %s", text)
    user = assert_user(user)
    chain = get_chain(user)

    with Session(engine) as session:
        try:
            if len(chain) == 0:
                samples = session.query(Sample).order_by(func.random()).limit(10)
                for sample in samples:
                    pred = predict(session, model, sample)
                    result = translate_result(pred)
                    chain.appendleft({"message": sample.text, "result": result})

            try:
                prev_msg = chain[0]["message"]
            except IndexError:
                prev_msg = ""

            if text and prev_msg != text:
                sample = get_sample(session, text)
                pred = predict(session, model, sample)
                result = translate_result(pred)
                chain.appendleft({"message": sample.text, "result": result})

            session.commit()
        except Exception:
            session.rollback()
            raise

    session_manager[user] = chain
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "data": chain, "user": user},
    )
