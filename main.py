"""
Metaculus AI Forecasting Benchmark — Ensemble Bot
=================================================

A competition bot for the Metaculus AI Forecasting Benchmark (FutureEval), built
on top of the official `forecasting-tools` framework. It implements the techniques
that the published Q4-2024 / Q2-2025 winner write-ups identified as the biggest
score levers, instead of the single-shot baseline of the stock template:

  1. ENSEMBLE / AGGREGATION (the single biggest lever in Metaculus' own analysis):
     every question is forecast multiple times and the results are aggregated with
     the framework's tested aggregator (median for binary, normalize for multiple
     choice, quantile-merge for numeric).
  2. MULTI-MODEL DIVERSITY: those runs are spread across *different* model families
     (GPT-4o + Claude), not one model repeated — winners averaged ~1.8 distinct
     models. Diverse models cancel each other's idiosyncratic errors.
  3. OUTSIDE-VIEW → INSIDE-VIEW PROMPTING: each forecast prompt forces a reference
     class / base-rate pass before the case-specific reasoning, then weights the
     status quo — the calibration habits good forecasters use.
  4. RESEARCH WITH FALLBACK: multi-step, source-grounded news research
     (AskNews DeepNews -> basic AskNews news -> search model -> LLM) so forecasts
     react to current evidence. DeepNews (AskNews' agentic deep-research endpoint)
     is used automatically whenever AskNews credentials are present; otherwise the
     bot degrades gracefully to a plain search model.

WHY IT'S ZERO-COST BY DEFAULT
-----------------------------
During the tournament Metaculus sponsors LLM access through its `metaculus/...`
proxy (keyed off your METACULUS_TOKEN). The default panel below uses two *different*
providers through that proxy, so you get a diverse ensemble for free — no OpenAI /
Anthropic / OpenRouter key required. Add your own keys only if you want newer or
stronger models (see FORECASTER_MODELS).

TUNING (all near the top of this file)
--------------------------------------
  FORECASTER_MODELS       which models form the ensemble panel
  RUNS_PER_MODEL          how many times each model forecasts each question
  MODEL_TEMPERATURE       sampling temperature (>0 gives the ensemble spread)
  MAX_CONCURRENT_LLM_CALLS global throttle so you don't trip provider rate limits

Total forecasts per question  =  len(FORECASTER_MODELS) * RUNS_PER_MODEL.
Default = 2 models * 2 runs = 4 aggregated forecasts per question.

Run:
  python main.py --mode test_questions     # smoke test on the bot-testing-area
  python main.py --mode tournament         # live AIB tournament + MiniBench
  python main.py --mode tournament --no-publish   # dry run, nothing submitted
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

import dotenv

from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    ForecastReport,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# ============================== CONFIG ========================================
# The ensemble panel — OpenRouter models verified (via _diag_llm.py) to actually
# work with this project's key. NOTE: the Metaculus free proxy returned "no
# allowance" for every model on this token, so we route ALL LLM calls through
# OpenRouter instead. The panel deliberately mixes THREE model families for
# diversity — that diversity is the whole point of the ensemble.
#
# All three are free / near-free on OpenRouter. Once you have OpenRouter credits you
# can swap in stronger models, e.g. "openrouter/openai/gpt-4o-mini" or
# "openrouter/anthropic/claude-3.5-haiku" — but RE-RUN `python _diag_llm.py` first to
# confirm your key/balance can call them (paid OpenAI/Anthropic 403'd on a $0 balance).
FORECASTER_MODELS: list[str] = [
    "openrouter/google/gemma-4-31b-it:free",   # Google Gemma family
    "openrouter/nex-agi/nex-n2-pro:free",      # Nex family
    "openrouter/inclusionai/ling-2.6-flash",   # InclusionAI Ling family (near-free)
]

# The framework's default parser/researcher point at OpenAI-via-OpenRouter, which a
# $0-balance key cannot call (403). Pin them to models this key CAN call, or parsing
# silently fails and every forecast dies.
PARSER_MODEL = "openrouter/google/gemma-4-31b-it:free"      # extracts the final %/option/percentiles
RESEARCHER_MODEL = "openrouter/google/gemma-4-31b-it:free"  # writes the research rundown (knowledge-based; no live web search on free tier)

RUNS_PER_MODEL = 1          # forecasts per model per question. =1 to conserve the OpenRouter free-tier daily quota (still a 3-model ensemble); raise once you have credits.
MODEL_TEMPERATURE = 0.4     # >0 so repeated runs differ; the spread is what we aggregate
REQUEST_TIMEOUT = 90        # seconds per LLM call
MAX_CONCURRENT_LLM_CALLS = 3  # global throttle. Low to stay under OpenRouter free-tier's ~20 req/min limit.
PARSER_VALIDATION_SAMPLES = 1  # text->structured parse robustness. =1 to save calls on the free tier; raise to 2 with credits.

# --- AskNews DeepNews deep research (used automatically IF AskNews creds are set) ---
# DeepNews is AskNews' agentic deep-research endpoint: it runs several rounds of
# search + reasoning and returns a cited report — much richer than a single news
# query. It costs AskNews credits per call (one call per question, shared across the
# whole ensemble), so the defaults below are deliberately conservative.
USE_ASKNEWS_DEEP_RESEARCH = True      # master switch; ignored unless AskNews creds exist
ASKNEWS_DEEP_RESEARCH_MODEL = "deepseek-basic"  # routes to a strong open-source model = cheapest.
#   Bump to "claude-sonnet-4-6", "gpt-5", or "o3" for stronger (pricier) research.
ASKNEWS_DEEP_SEARCH_DEPTH = 2         # rounds of search->reason; higher = more thorough, slower, pricier
ASKNEWS_DEEP_MAX_DEPTH = 4            # hard ceiling on research depth
ASKNEWS_DEEP_SOURCES = ["asknews"]    # add "google","x","wiki" for broader coverage (may need a premium plan)
ASKNEWS_DEEP_FILTER_PARAMS = None     # e.g. {"premium": True} to use premium sources (needs the plan)
ASKNEWS_DEEP_RESEARCH_TIMEOUT = 300   # seconds; abandon a single deep-research call after this
MAX_CONCURRENT_RESEARCH_CALLS = 2     # deepnews is slow + rate-limited; keep this low
# ==============================================================================


def check_environment() -> None:
    """Fail fast with a clear message if the bot can't possibly run."""
    token = os.getenv("METACULUS_TOKEN")
    if not token or token.strip() in {"", "REPLACE_ME"}:
        print(
            "[FATAL] METACULUS_TOKEN is not set.\n"
            "        Create one at https://www.metaculus.com/futureeval/participate/\n"
            "        then put it in a local .env file or a GitHub Actions secret.",
            file=sys.stderr,
        )
        sys.exit(1)

    byok = any(
        os.getenv(k)
        for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not byok:
        logger.info(
            "No personal LLM key found — using the sponsored Metaculus proxy "
            "(free during the tournament). This is expected and fine."
        )


class EnsembleBot(ForecastBot):
    """
    Multi-model, multi-run ensemble forecaster.

    Design note on how this plugs into forecasting-tools:
    We do ALL ensembling inside each `_run_forecast_on_*` method and return a single
    aggregated `ReasonedPrediction`. So the bot is configured with
    `predictions_per_research_report=1` (one research pass, one aggregated answer per
    question); the fan-out happens internally in `_run_ensemble`. This keeps every
    question's ensemble self-contained and balanced across the panel, and reuses the
    framework's own `_aggregate_predictions` for each question type.
    """

    @property
    def panel(self) -> list[GeneralLlm]:
        """Lazily build (and cache) the panel of GeneralLlm forecasters."""
        if getattr(self, "_panel", None) is None:
            self._panel = [
                GeneralLlm(
                    model=name,
                    temperature=MODEL_TEMPERATURE,
                    timeout=REQUEST_TIMEOUT,
                    allowed_tries=2,
                )
                for name in FORECASTER_MODELS
            ]
            logger.info(
                "Forecaster panel: %s  (x%d runs each = %d forecasts/question)",
                [m.model for m in self._panel],
                RUNS_PER_MODEL,
                len(self._panel) * RUNS_PER_MODEL,
            )
        return self._panel

    async def _invoke(self, model: GeneralLlm, prompt: str) -> str:
        """
        Rate-limited reasoning call. The semaphore is created lazily here, inside the
        running event loop, so it never gets reused across two different event loops
        (which would raise "bound to a different event loop").
        """
        if getattr(self, "_limiter", None) is None:
            self._limiter = asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)
        async with self._limiter:
            return await model.invoke(prompt)

    # ------------------------------- RESEARCH --------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Gather grounding research ONCE per question (shared across the whole
        ensemble). Strategy, best first, each degrading gracefully into the next:
          1. AskNews DeepNews  — agentic multi-round deep research (if AskNews creds)
          2. the framework's configured researcher — basic AskNews news, a search
             model, or the sponsored Metaculus search proxy
          3. ""               — research must never sink the forecast
        The prompt asks for an evidence rundown plus a base-rate / reference-class
        pass so the forecast prompts get an outside-view signal, not just headlines.
        """
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster. You do NOT give a
            probability yourself. Produce a concise, source-grounded rundown that the
            forecaster can act on.

            Question:
            {question.question_text}

            Resolution criteria (how this resolves):
            {question.resolution_criteria}

            {question.fine_print or ""}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            In your rundown cover, with concrete dates/numbers where possible:
            1. The most relevant recent developments and the current status quo.
            2. A reference class / base rate: how often do situations like this resolve
               Yes vs No historically? What is the outside-view prior?
            3. The key drivers that could move the outcome before resolution, and the
               main uncertainties.
            4. What current evidence implies about the likely resolution.
            """
        )

        # 1. Preferred: AskNews DeepNews deep research (only if creds are configured).
        if USE_ASKNEWS_DEEP_RESEARCH and self._asknews_configured():
            try:
                research = await self._deep_research(prompt)
                if research and research.strip():
                    logger.info(
                        "Research (AskNews DeepNews) for %s:\n%s",
                        question.page_url, research,
                    )
                    return research
                logger.warning(
                    "AskNews DeepNews returned empty research for %s; falling back.",
                    question.page_url,
                )
            except Exception as exc:  # deepnews must never sink the forecast
                logger.warning(
                    "AskNews DeepNews failed for %s (%r); falling back to the "
                    "default researcher.",
                    question.page_url, exc,
                )

        # 2. Fallback: whatever researcher the framework configured from your env.
        research = await self._fallback_research(prompt)
        logger.info("Research for %s:\n%s", question.page_url, research)
        return research

    @staticmethod
    def _asknews_configured() -> bool:
        """True if AskNews credentials (OAuth pair or API key) are present in env."""
        return bool(
            (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
            or os.getenv("ASKNEWS_API_KEY")
        )

    def _get_asknews(self) -> AskNewsSearcher:
        """Lazily build (and cache) one AskNewsSearcher; safe to reuse across calls."""
        if getattr(self, "_asknews", None) is None:
            self._asknews = AskNewsSearcher()
        return self._asknews

    async def _deep_research(self, prompt: str) -> str:
        """
        Run one AskNews DeepNews call — throttled and time-bounded. The semaphore is
        created lazily inside the running loop (so it's never shared across two event
        loops), and a timeout stops a single deep-research call from hanging the run.
        """
        if getattr(self, "_research_limiter", None) is None:
            self._research_limiter = asyncio.Semaphore(MAX_CONCURRENT_RESEARCH_CALLS)
        async with self._research_limiter:
            return await asyncio.wait_for(
                self._get_asknews().get_formatted_deep_research(
                    prompt,
                    model=ASKNEWS_DEEP_RESEARCH_MODEL,
                    search_depth=ASKNEWS_DEEP_SEARCH_DEPTH,
                    max_depth=ASKNEWS_DEEP_MAX_DEPTH,
                    sources=ASKNEWS_DEEP_SOURCES,
                    filter_params=ASKNEWS_DEEP_FILTER_PARAMS,
                ),
                timeout=ASKNEWS_DEEP_RESEARCH_TIMEOUT,
            )

    async def _fallback_research(self, prompt: str) -> str:
        """The framework's configured researcher, with the same never-crash guarantee."""
        researcher = self.get_llm("researcher")
        try:
            if isinstance(researcher, GeneralLlm):
                return await researcher.invoke(prompt)
            if isinstance(researcher, str) and researcher.startswith("asknews"):
                return await self._get_asknews().call_preconfigured_version(
                    researcher, prompt
                )
            if isinstance(researcher, str) and researcher.startswith("smart-searcher"):
                return await SmartSearcher(
                    model=researcher.removeprefix("smart-searcher/"),
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                ).invoke(prompt)
            if not researcher or researcher in {"None", "no_research"}:
                return ""
            return await self.get_llm("researcher", "llm").invoke(prompt)
        except Exception as exc:  # research must never sink the whole forecast
            logger.warning("Fallback research failed: %r", exc)
            return ""

    # ---------------------------- ENSEMBLE CORE ------------------------------
    async def _run_ensemble(self, question: MetaculusQuestion, predict_once) -> ReasonedPrediction:
        """
        Run `predict_once(model)` for every (model, run) in the panel, drop failures,
        aggregate the survivors with the framework's tested aggregator, and attach a
        transparent rationale listing every sub-forecast.

        `predict_once` is an async callable returning a 4-tuple:
            (prediction_value, model_label, pretty_value, reasoning_text)
        where prediction_value is float | PredictedOptionList | NumericDistribution.
        """
        tasks = [
            predict_once(model)
            for model in self.panel
            for _ in range(RUNS_PER_MODEL)
        ]
        settled = await asyncio.gather(*tasks, return_exceptions=True)

        predictions, notes, errors = [], [], []
        for result in settled:
            if isinstance(result, BaseException):
                errors.append(result)
                continue
            value, label, pretty, reasoning = result
            predictions.append(value)
            notes.append((label, pretty, reasoning))

        if not predictions:
            raise RuntimeError(
                f"All {len(tasks)} ensemble forecasts failed for {question.page_url}. "
                f"First error: {errors[0] if errors else 'n/a'}"
            )
        if errors:
            logger.warning(
                "%d/%d sub-forecasts failed for %s",
                len(errors), len(tasks), question.page_url,
            )
            # Log each error so a genuine code bug (vs. an LLM/format hiccup) is
            # visible instead of being silently absorbed by return_exceptions=True.
            for err in errors:
                logger.warning("  sub-forecast error: %r", err)

        aggregated = await self._aggregate_predictions(predictions, question)
        logger.info(
            "Aggregated %s -> %s",
            question.page_url, self._pretty_prediction(aggregated),
        )
        return ReasonedPrediction(
            prediction_value=aggregated,
            reasoning=self._format_ensemble_reasoning(aggregated, notes, len(errors)),
        )

    @staticmethod
    def _pretty_prediction(prediction) -> str:
        if isinstance(prediction, float):
            return f"{prediction:.1%}"
        if isinstance(prediction, NumericDistribution):
            return str(prediction.declared_percentiles)
        return str(prediction)

    def _format_ensemble_reasoning(self, aggregated, notes, num_failed: int) -> str:
        lines = [
            "# Ensemble forecast",
            f"Aggregated {len(notes)} sub-forecast(s) from a "
            f"{len(self.panel)}-model panel"
            + (f" ({num_failed} failed)." if num_failed else "."),
            f"**Final (aggregated): {self._pretty_prediction(aggregated)}**",
            "",
            "## Individual model forecasts",
        ]
        for i, (label, pretty, reasoning) in enumerate(notes, 1):
            lines.append(f"\n### {i}. {label} -> {pretty}")
            text = reasoning.strip()
            # Cap each sub-rationale so the published comment stays well under
            # Metaculus' comment-size limit even with a large panel / many runs.
            lines.append(text if len(text) <= 4000 else text[:4000] + " …[truncated]")
        return "\n".join(lines)

    # ------------------------------- BINARY ----------------------------------
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Forecast the probability that this
            question resolves Yes.

            Question:
            {question.question_text}

            Background:
            {question.background_info or "No background provided."}

            Resolution criteria (not yet satisfied):
            {question.resolution_criteria}

            {question.fine_print or ""}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Reason in this order, writing each part out:
            (a) Time left until the outcome is known.
            (b) OUTSIDE VIEW: the reference class and its base rate — absent specific
                information, how often do situations like this resolve Yes?
            (c) The status quo outcome if nothing changed.
            (d) INSIDE VIEW: the case-specific evidence from the research that moves you
                off the base rate, and in which direction.
            (e) A brief scenario that yields No.
            (f) A brief scenario that yields Yes.
            (g) Synthesis: combine the outside and inside views. Good forecasters put
                extra weight on the status quo because the world changes slowly, and
                avoid overconfident 0%/100% answers.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100.
            """
        )

        async def predict_once(model: GeneralLlm):
            reasoning = await self._invoke(model, prompt)
            parsed: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=PARSER_VALIDATION_SAMPLES,
            )
            probability = max(0.01, min(0.99, parsed.prediction_in_decimal))
            return probability, model.model, f"{probability:.1%}", reasoning

        return await self._run_ensemble(question, predict_once)

    # --------------------------- MULTIPLE CHOICE -----------------------------
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Assign a probability to each option.

            Question:
            {question.question_text}

            Options (use these exact names): {question.options}

            Background:
            {question.background_info or "No background provided."}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print or ""}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Reason in this order, writing each part out:
            (a) Time left until the outcome is known.
            (b) OUTSIDE VIEW: base rates across the options — what is the prior
                distribution absent specific information?
            (c) The status quo outcome if nothing changed.
            (d) INSIDE VIEW: case-specific evidence and how it reshapes the distribution.
            (e) An unexpected-outcome scenario.

            Good forecasters (1) weight the status quo, and (2) leave moderate
            probability on every option to account for surprises — never assign 0%.

            The last thing you write is the final probability for each option, in the
            order {question.options}, as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure each option name is exactly one of: {question.options}
            Strip any leading "Option" label that isn't part of the real name.
            Include every option, even ones at 0% (list them, don't skip them).
            """
        )

        async def predict_once(model: GeneralLlm):
            reasoning = await self._invoke(model, prompt)
            parsed: PredictedOptionList = await structure_output(
                text_to_structure=reasoning,
                output_type=PredictedOptionList,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=PARSER_VALIDATION_SAMPLES,
                additional_instructions=parsing_instructions,
            )
            return parsed, model.model, str(parsed), reasoning

        return await self._run_ensemble(question, predict_once)

    # ------------------------------- NUMERIC ---------------------------------
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_msg, lower_msg = self._bound_messages(question)
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Produce a calibrated probability
            distribution for the answer.

            Question:
            {question.question_text}

            Background:
            {question.background_info or "No background provided."}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print or ""}

            Units for the answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (infer it)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_msg}
            {upper_msg}

            Formatting rules:
            - Answer in the requested units. Never use scientific notation.
            - Percentiles must strictly increase: P10 < P20 < P40 < P60 < P80 < P90.

            Reason in this order, writing each part out:
            (a) Time left until the outcome is known.
            (b) OUTSIDE VIEW: the historical base rate / typical range for this kind of
                quantity.
            (c) The outcome if nothing changed, and if the current trend continued.
            (d) Expectations of experts and markets.
            (e) INSIDE VIEW: case-specific evidence from the research.
            (f) A low-tail scenario and a high-tail scenario.

            Good forecasters are humble: set WIDE 90/10 intervals to cover unknown
            unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest)
            "
            """
        )
        parsing_instructions = clean_indents(
            f"""
            The text is a forecast distribution for: "{question.question_text}".
            - Give the percentile values in the correct units: {question.unit_of_measure or "(infer the units)"}.
            - Convert any scientific notation into plain numbers.
            - If the text only gives a single value (no percentiles), indicate that the
              distribution is not explicitly given rather than inventing one.
            """
        )

        async def predict_once(model: GeneralLlm):
            reasoning = await self._invoke(model, prompt)
            percentiles: list[Percentile] = await structure_output(
                reasoning,
                list[Percentile],
                model=self.get_llm("parser", "llm"),
                num_validation_samples=PARSER_VALIDATION_SAMPLES,
                additional_instructions=parsing_instructions,
            )
            distribution = NumericDistribution.from_question(percentiles, question)
            return (
                distribution,
                model.model,
                str(distribution.declared_percentiles),
                reasoning,
            )

        return await self._run_ensemble(question, predict_once)

    @staticmethod
    def _bound_messages(question: NumericQuestion) -> tuple[str, str]:
        upper = (
            question.nominal_upper_bound
            if question.nominal_upper_bound is not None
            else question.upper_bound
        )
        lower = (
            question.nominal_lower_bound
            if question.nominal_lower_bound is not None
            else question.lower_bound
        )
        unit = question.unit_of_measure or ""
        upper_msg = (
            f"The question creator thinks the number is likely not higher than {upper} {unit}."
            if question.open_upper_bound
            else f"The outcome can not be higher than {upper} {unit}."
        )
        lower_msg = (
            f"The question creator thinks the number is likely not lower than {lower} {unit}."
            if question.open_lower_bound
            else f"The outcome can not be lower than {lower} {unit}."
        )
        return upper_msg, lower_msg


def _print_summary(reports, will_publish: bool) -> None:
    valid = [r for r in reports if isinstance(r, ForecastReport)]
    failed = [r for r in reports if isinstance(r, BaseException)]
    bar = "=" * 70
    print(f"\n{bar}")
    if not reports:
        print("No new questions to forecast this run.")
    else:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"{verb.capitalize()} {len(valid)} forecast(s); {len(failed)} failed.")
        for r in valid:
            print(f"  OK  {r.question.page_url}")
        for e in failed:
            msg = str(e)
            print(f"  ERR {type(e).__name__}: {msg[:180]}")
    print(f"{bar}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament = live AIB + MiniBench)",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Dry run: compute forecasts but do NOT submit them to Metaculus.",
    )
    args = parser.parse_args()

    check_environment()
    publish = not args.no_publish
    print(f"Mode={args.mode}  publish={'yes' if publish else 'no (dry run)'}")

    bot = EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # ensemble fan-out happens inside the bot
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,  # skip the unused summary call (we forecast on full research)
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            # Pin every support role to a model this key can actually call. The
            # framework defaults point at OpenAI-via-OpenRouter (403 on a $0 balance),
            # which would silently break the parser and sink every forecast.
            "default": GeneralLlm(
                model=FORECASTER_MODELS[0],
                temperature=MODEL_TEMPERATURE,
                timeout=REQUEST_TIMEOUT,
            ),
            "parser": GeneralLlm(
                model=PARSER_MODEL, temperature=0, timeout=REQUEST_TIMEOUT
            ),
            "researcher": GeneralLlm(
                model=RESEARCHER_MODEL, temperature=0.1, timeout=REQUEST_TIMEOUT
            ),
        },
    )

    client = MetaculusClient()
    if args.mode == "tournament":

        async def _forecast_tournament():
            # Run both tournaments inside ONE event loop so loop-bound primitives
            # (the rate-limiter semaphore) aren't shared across event loops.
            aib = await bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
            minibench = await bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
            return aib + minibench

        reports = asyncio.run(_forecast_tournament())
    elif args.mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    else:  # test_questions
        bot.skip_previously_forecasted_questions = False
        reports = asyncio.run(
            bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
        )

    if hasattr(bot, "log_report_summary"):
        try:
            bot.log_report_summary(reports)
        except Exception:  # summary logging must never crash the run
            pass
    _print_summary(reports, publish)
