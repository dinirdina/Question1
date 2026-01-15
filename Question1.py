import numpy as np
import pandas as pd
import streamlit as st

POP_SIZE = 300
CHROM_LEN = 80
GENERATIONS = 50
TARGET_ONES = 40
MAX_FITNESS = 80


CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
ELITISM = 2


def fitness(ind: np.ndarray) -> float:
    ones = int(np.sum(ind))

    return float(MAX_FITNESS - abs(ones - TARGET_ONES))

# -------------------- GA Helpers --------------------
def init_population(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(POP_SIZE, CHROM_LEN), dtype=np.int8)

def evaluate(pop: np.ndarray) -> np.ndarray:
    return np.array([fitness(ind) for ind in pop], dtype=float)

def tournament_selection(fit: np.ndarray, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fit.size, size=TOURNAMENT_K)
    return int(idxs[np.argmax(fit[idxs])])

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    point = int(rng.integers(1, CHROM_LEN))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    mask = rng.random(y.shape) < MUTATION_RATE
    y[mask] = 1 - y[mask]
    return y

# -------------------- GA Runner --------------------
def run_ga(seed: int, live: bool):
    rng = np.random.default_rng(seed)
    pop = init_population(rng)
    fit = evaluate(pop)

    history_best, history_avg, history_worst = [], [], []

    chart_area = st.empty()
    best_area = st.empty()

    for gen in range(GENERATIONS):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])

        history_best.append(best_fit)
        history_avg.append(float(np.mean(fit)))
        history_worst.append(float(np.min(fit)))

        if live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{GENERATIONS} — Best fitness: **{best_fit:.2f}**")

        # Elitism
        E = min(ELITISM, POP_SIZE)
        elite_idx = np.argpartition(fit, -E)[-E:]
        elites = pop[elite_idx].copy()

        # Build next population
        next_pop = []
        while len(next_pop) < POP_SIZE - E:
            p1 = pop[tournament_selection(fit, rng)]
            p2 = pop[tournament_selection(fit, rng)]

            if rng.random() < CROSSOVER_RATE:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, rng)
            c2 = bit_mutation(c2, rng)

            next_pop.append(c1)
            if len(next_pop) < POP_SIZE - E:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop), elites])
        fit = evaluate(pop)

    # Final best
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    history_df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
    return best, best_fit, history_df, pop, fit

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="GA Lab Test - Bit Pattern", page_icon="🧬", layout="wide")
st.title("GA Lab Test: Bit Pattern Generator")
st.caption("Fixed parameters as required by the question (Population=300, Length=80, Generations=50, peak at 40 ones).")

with st.sidebar:
    st.header("Fixed Parameters (Lab)")
    st.write(f"Population: **{POP_SIZE}**")
    st.write(f"Chromosome Length: **{CHROM_LEN}**")
    st.write(f"Generations: **{GENERATIONS}**")
    st.write(f"Fitness peak (ones): **{TARGET_ONES}**")
    st.write(f"Max fitness: **{MAX_FITNESS}**")

    st.header("Run Settings")
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    live = st.checkbox("Live chart while running", value=True)

if "_final_pop" not in st.session_state:
    st.session_state["_final_pop"] = None
    st.session_state["_final_fit"] = None

left, right = st.columns([1, 1])

with left:
    if st.button("Run GA", type="primary"):
        best, best_fit, history, final_pop, final_fit = run_ga(int(seed), bool(live))

        # store for table view
        st.session_state["_final_pop"] = final_pop
        st.session_state["_final_fit"] = final_fit

        st.subheader("Fitness Over Generations")
        st.line_chart(history)

        st.subheader("Best Solution")
        ones = int(np.sum(best))
        bitstring = "".join(map(str, best.astype(int).tolist()))

        st.write(f"Best fitness: **{best_fit:.2f} / {MAX_FITNESS}**")
        st.write(f"Number of ones: **{ones}** (target is **{TARGET_ONES}**)")
        st.code(bitstring, language="text")

        st.info(
            "Interpretation tip: fitness is highest when the chromosome contains exactly 40 ones. "
            "GA should converge toward individuals near 40 ones; best fitness should approach 80."
        )

with right:
    st.subheader("Population Snapshot (final)")
    st.caption("Shows first 20 individuals with fitness")

    if st.button("Show final population table"):
        pop = st.session_state["_final_pop"]
        fit = st.session_state["_final_fit"]

        if pop is None or fit is None:
            st.warning("Run GA first.")
        else:
            nshow = min(20, pop.shape[0])
            df = pd.DataFrame(pop[:nshow])
            df["ones"] = pop[:nshow].sum(axis=1)
            df["fitness"] = fit[:nshow]
            st.dataframe(df, use_container_width=True)
