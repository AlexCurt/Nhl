"""NHL Buteurs - outil perso de detection de value (V1)

Modele : buts attendus = tirs/match x % de reussite x forme x gardien adverse
x defense adverse x domicile/exterieur. Proba de marquer = 1 - exp(-buts attendus).
Les cotes Winamax sont saisies a la main dans le tableau.
"""
import base64
import io
import math
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

API = "https://api-web.nhle.com/v1"
ODDS_API = "https://api.the-odds-api.com/v4"

# --- Parametres du modele (a ajuster) ---
LEAGUE_SV = 0.905        # % d'arrets moyen NHL
LEAGUE_SH_PCT = 0.10     # % de reussite moyen des patineurs
LEAGUE_SPG = 2.0         # tirs/match moyens d'un patineur offensif
W_PRIOR = 0.6            # poids d'un match de la saison precedente
HOME_BOOST = 1.02        # petit bonus domicile (0.98 en exterieur)
KELLY_FRACTION = 0.25    # on mise 1/4 de Kelly (prudent)

st.set_page_config(page_title="NHL Buteurs", page_icon="🏒", layout="wide")


# ---------------------------------------------------------------- acces
def check_password():
    try:
        expected = st.secrets.get("APP_PASSWORD")
    except Exception:
        expected = None
    if not expected:
        st.error("Mot de passe non configuré : ajoute APP_PASSWORD dans Settings > Secrets.")
        st.stop()
    if st.session_state.get("auth"):
        return
    pwd = st.text_input("Mot de passe", type="password")
    if pwd:
        if pwd == expected:
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    st.stop()


# ---------------------------------------------------------------- donnees
@st.cache_data(ttl=1800, show_spinner=False)
def get_json(path):
    r = requests.get(f"{API}{path}", timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def season_ids(day):
    y = day.year if day.month >= 9 else day.year - 1
    return f"{y}{y + 1}", f"{y - 1}{y}"


def name_of(p):
    def val(x):
        return x.get("default", "") if isinstance(x, dict) else (x or "")
    return f"{val(p.get('firstName'))} {val(p.get('lastName'))}".strip()


def get_games(day):
    data = get_json(f"/schedule/{day.isoformat()}")
    for w in data.get("gameWeek", []):
        if w.get("date") == day.isoformat():
            return [g for g in w.get("games", []) if g.get("gameType") in (2, 3)]
    return []


def load_team(team, season):
    try:
        d = get_json(f"/club-stats/{team}/{season}/2")
    except Exception:
        return [], []
    return d.get("skaters", []), d.get("goalies", [])


def league_context():
    """Buts encaisses par match de chaque equipe + moyenne ligue."""
    try:
        rows = get_json("/standings/now").get("standings", [])
    except Exception:
        return {}, 3.0
    ga, tot_g, tot_gp = {}, 0, 0
    for t in rows:
        ab = t.get("teamAbbrev", {}).get("default")
        g, gp = t.get("goalAgainst", 0), t.get("gamesPlayed", 0)
        if ab:
            ga[ab] = (g, gp)
            tot_g += g
            tot_gp += gp
    return ga, (tot_g / tot_gp if tot_gp else 3.0)


# ---------------------------------------------------------------- cotes US
def american_to_decimal(a):
    """+150 -> 2.50 ; -200 -> 1.50"""
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def norm(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c for c in s if c.isalnum() or c == " ").split())


def odds_key():
    try:
        return st.secrets.get("ODDS_API_KEY")
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_us_odds(away_names, home_names, key):
    """Cotes 'buteur a tout moment' des bookmakers US, en decimal.
    Retourne ({nom_normalise: {'avg','best','n'}}, credits_restants)."""
    r = requests.get(f"{ODDS_API}/sports/icehockey_nhl/events", params={"apiKey": key}, timeout=20)
    r.raise_for_status()
    ev = None
    for e in r.json():
        h, a = e.get("home_team", "").lower(), e.get("away_team", "").lower()
        if any(n.lower() in h for n in home_names if n) and any(n.lower() in a for n in away_names if n):
            ev = e
            break
    if not ev:
        return {}, None
    r = requests.get(
        f"{ODDS_API}/sports/icehockey_nhl/events/{ev['id']}/odds",
        params={"apiKey": key, "regions": "us", "markets": "player_goal_scorer_anytime", "oddsFormat": "decimal"},
        timeout=20,
    )
    r.raise_for_status()
    prices = {}
    for bk in r.json().get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "player_goal_scorer_anytime":
                continue
            for o in m.get("outcomes", []):
                if o.get("name") in ("Yes", "Over") and o.get("description") and o.get("price"):
                    prices.setdefault(norm(o["description"]), []).append(float(o["price"]))
    out = {k: {"avg": sum(v) / len(v), "best": max(v), "n": len(v)} for k, v in prices.items()}
    return out, r.headers.get("x-requests-remaining")


# ---------------------------------------------------------------- modele
def goalie_sv(g):
    sa = g.get("shotsAgainst") or 0
    saves = g.get("saves")
    if saves is None:
        saves = (g.get("savePercentage") or LEAGUE_SV) * sa
    return (saves + LEAGUE_SV * 500) / (sa + 500)   # regression vers la moyenne


def def_factor(ga_gp, league_pg):
    ga, gp = ga_gp
    if not gp:
        return 1.0
    w = gp / (gp + 25)
    return 1 + 0.5 * w * ((ga / gp) / league_pg - 1)


def profile(pid, prior):
    """Saison precedente + 5 derniers matchs via la fiche joueur."""
    gp_p = sh_p = g_p = 0
    l5 = []
    try:
        land = get_json(f"/player/{pid}/landing")
    except Exception:
        return gp_p, sh_p, g_p, l5
    for s in land.get("seasonTotals", []):
        if (str(s.get("season")) == prior and s.get("leagueAbbrev") == "NHL"
                and s.get("gameTypeId") == 2 and "shots" in s):
            gp_p += s.get("gamesPlayed", 0)
            sh_p += s.get("shots", 0)
            g_p += s.get("goals", 0)
    l5 = [x["shots"] for x in land.get("last5Games", []) if x.get("shots") is not None]
    return gp_p, sh_p, g_p, l5


def build_rows(team, skaters, prior, n, is_home, opp_sv, opp_def):
    cand = [s for s in skaters if (s.get("gamesPlayed") or 0) > 0]
    cand.sort(key=lambda s: s.get("shots", 0), reverse=True)
    rows = []
    for s in cand[:n]:
        gp_c, sh_c, g_c = s.get("gamesPlayed", 0), s.get("shots", 0), s.get("goals", 0)
        gp_p, sh_p, g_p, l5 = profile(s["playerId"], prior)

        gp = gp_c + W_PRIOR * gp_p
        spg = (sh_c + W_PRIOR * sh_p + LEAGUE_SPG * 5) / (gp + 5)
        shpct = (g_c + W_PRIOR * g_p + LEAGUE_SH_PCT * 100) / (sh_c + W_PRIOR * sh_p + 100)

        form = 1.0
        if l5 and spg > 0:
            form = min(1.10, max(0.90, 1 + 0.25 * ((sum(l5) / len(l5)) / spg - 1)))

        goalie = (1 - opp_sv) / (1 - LEAGUE_SV)
        home = HOME_BOOST if is_home else 2 - HOME_BOOST
        xg = spg * shpct * form * goalie * opp_def * home
        p = 1 - math.exp(-xg)

        rows.append({
            "Joue": True,
            "_id": s["playerId"],
            "Joueur": name_of(s),
            "Équipe": team,
            "Tirs/match": round(spg, 2),
            "% réussite": round(shpct * 100, 1),
            "Forme": round(form, 2),
            "Buts attendus": round(xg, 3),
            "Proba but %": round(p * 100, 1),
            "Cote juste": round(1 / p, 2) if p > 0 else None,
            "Cote Winamax": None,
        })
    return rows


# ---------------------------------------------------------------- sauvegarde GitHub (branche 'data')
DATA_BRANCH = "data"


def gh_conf():
    try:
        return st.secrets.get("GITHUB_TOKEN"), st.secrets.get("GITHUB_REPO")
    except Exception:
        return None, None


def _gh(token, raw=False):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json"}


def gh_ensure_branch(token, repo):
    base = f"https://api.github.com/repos/{repo}"
    r = requests.get(f"{base}/git/ref/heads/{DATA_BRANCH}", headers=_gh(token), timeout=20)
    if r.status_code == 200:
        return
    r = requests.get(f"{base}/git/ref/heads/main", headers=_gh(token), timeout=20)
    r.raise_for_status()
    sha = r.json()["object"]["sha"]
    r = requests.post(f"{base}/git/refs", headers=_gh(token), timeout=20,
                      json={"ref": f"refs/heads/{DATA_BRANCH}", "sha": sha})
    r.raise_for_status()


def gh_save(path, text, token, repo):
    gh_ensure_branch(token, repo)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers=_gh(token), params={"ref": DATA_BRANCH}, timeout=20)
    body = {"message": f"predictions {path}", "branch": DATA_BRANCH,
            "content": base64.b64encode(text.encode("utf-8")).decode()}
    if r.status_code == 200:
        body["sha"] = r.json()["sha"]
    r = requests.put(url, headers=_gh(token), json=body, timeout=20)
    r.raise_for_status()


@st.cache_data(ttl=300, show_spinner=False)
def gh_load_predictions(token, repo):
    url = f"https://api.github.com/repos/{repo}/contents/predictions"
    r = requests.get(url, headers=_gh(token), params={"ref": DATA_BRANCH}, timeout=20)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    frames = []
    for f in r.json():
        if f["name"].endswith(".csv"):
            rr = requests.get(f["url"], headers=_gh(token, raw=True), params={"ref": DATA_BRANCH}, timeout=20)
            rr.raise_for_status()
            frames.append(pd.read_csv(io.StringIO(rr.text)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def boxscore_goals(game_id):
    """{playerId: buts} si le match est termine, sinon None."""
    d = get_json(f"/gamecenter/{game_id}/boxscore")
    if d.get("gameState") not in ("OFF", "FINAL"):
        return None
    out = {}
    for side in ("awayTeam", "homeTeam"):
        grp = d.get("playerByGameStats", {}).get(side, {})
        for pos in ("forwards", "defense"):
            for pl in grp.get(pos, []):
                out[pl["playerId"]] = pl.get("goals", 0)
    return out


def render_bilan():
    token, repo = gh_conf()
    if not (token and repo):
        st.info("Le suivi nécessite GITHUB_TOKEN et GITHUB_REPO dans les Secrets (voir les instructions).")
        return
    try:
        df = gh_load_predictions(token, repo)
    except Exception as e:
        st.warning(f"Lecture des prédictions impossible : {e}")
        return
    if df.empty:
        st.info("Aucune prédiction enregistrée pour l'instant. Utilise le bouton « Enregistrer » avant les matchs.")
        return
    goals = {}
    for gid in df["game_id"].unique():
        try:
            goals[gid] = boxscore_goals(int(gid))
        except Exception:
            goals[gid] = None
    df["but"] = [
        (1 if (goals.get(r.game_id) or {}).get(int(r.player_id), 0) > 0 else 0)
        if goals.get(r.game_id) is not None else None
        for r in df.itertuples()
    ]
    done = df.dropna(subset=["but"]).copy()
    st.write(f"{len(df)} prédictions enregistrées, {len(done)} sur des matchs terminés.")
    if done.empty:
        return
    y = done["but"].astype(float)

    def brier(col, sub, yy):
        return float(((sub[col] / 100 - yy) ** 2).mean())

    mk = done.dropna(subset=["p_mkt"])
    rows = [{"Source": "Modèle seul", "Brier": round(brier("p_model", mk, mk["but"]), 4)},
            {"Source": "Marché US corrigé", "Brier": round(brier("p_mkt", mk, mk["but"]), 4)},
            {"Source": "Proba finale", "Brier": round(brier("p_final", mk, mk["but"]), 4)}]
    st.markdown("**Score de Brier** (plus bas = meilleur), sur les joueurs avec cote US")
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    done["Tranche"] = pd.cut(done["p_final"], [0, 15, 25, 35, 100], labels=["<15 %", "15-25 %", "25-35 %", ">35 %"])
    cal = done.groupby("Tranche", observed=True).agg(
        Prédictions=("but", "size"), Proba_moyenne=("p_final", "mean"), Taux_réel=("but", "mean"))
    cal["Taux_réel"] = (cal["Taux_réel"] * 100).round(1)
    cal["Proba_moyenne"] = cal["Proba_moyenne"].round(1)
    st.markdown("**Calibration** : la proba annoncée doit ressembler au taux réel")
    st.dataframe(cal)
    if len(done) < 300:
        st.caption("Échantillon encore petit (< 300) : ne tire aucune conclusion ferme avant plusieurs semaines.")


# ---------------------------------------------------------------- interface
def render_game(g, season, prior, ga_map, league_pg, n, bankroll, margin, w_model, edge, day):
    away, home = g["awayTeam"]["abbrev"], g["homeTeam"]["abbrev"]
    try:
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        st.caption("Coup d'envoi : " + start.astimezone(ZoneInfo("Europe/Paris")).strftime("%d/%m %H:%M (heure de Paris)"))
    except Exception:
        pass

    data = {}
    for t in (away, home):
        sk, gl = load_team(t, season)
        gl = sorted(gl, key=lambda x: x.get("gamesStarted", 0), reverse=True)
        data[t] = (sk, gl)

    # choix du gardien adverse (par defaut : celui qui a le plus demarre)
    sv = {}
    cols = st.columns(2)
    for col, t in zip(cols, (away, home)):
        gl = data[t][1]
        with col:
            if gl:
                labels = [f"{name_of(x)} ({goalie_sv(x):.3f})" for x in gl]
                pick = st.selectbox(f"Gardien {t}", labels, key=f"g_{g['id']}_{t}")
                sv[t] = goalie_sv(gl[labels.index(pick)])
            else:
                st.caption(f"Gardien {t} : pas de données, moyenne ligue utilisée")
                sv[t] = LEAGUE_SV

    rows = []
    for t, opp, is_home in ((away, home, False), (home, away, True)):
        rows += build_rows(
            t, data[t][0], prior, n, is_home,
            opp_sv=sv[opp], opp_def=def_factor(ga_map.get(opp, (0, 0)), league_pg),
        )
    if not rows:
        st.warning("Pas de stats joueurs disponibles pour ce match.")
        return

    df = pd.DataFrame(rows).sort_values("Proba but %", ascending=False).reset_index(drop=True)

    # ---- cotes americaines (optionnel)
    key = odds_key()
    us = {}
    if key:
        def names(t):
            return tuple((g[t].get(k) or {}).get("default", "") for k in ("commonName", "placeName"))
        try:
            us, left = fetch_us_odds(names("awayTeam"), names("homeTeam"), key)
            if us:
                st.caption(f"Cotes US chargées ({len(us)} joueurs). Crédits API restants : {left}")
            else:
                st.caption("Cotes US : aucun marché buteur trouvé pour ce match (pas encore publié ?).")
        except Exception as e:
            st.warning(f"Cotes US indisponibles : {e}")
    else:
        st.caption("Ajoute ODDS_API_KEY dans Secrets pour afficher les cotes américaines.")
    df["Cote US moy."] = df["Joueur"].map(lambda x: round(us[norm(x)]["avg"], 2) if norm(x) in us else None)
    df["Meilleure US"] = df["Joueur"].map(lambda x: round(us[norm(x)]["best"], 2) if norm(x) in us else None)

    # ---- proba marche corrigee de la marge, puis proba finale (melange modele / marche)
    df["Proba marché %"] = (100 / df["Cote US moy."] / (1 + margin)).round(1)
    blend = w_model * df["Proba but %"] + (1 - w_model) * df["Proba marché %"]
    df["Proba finale %"] = blend.where(df["Proba marché %"].notna(), df["Proba but %"]).round(1)
    df["Cote juste"] = (100 / df["Proba finale %"]).round(2)
    df["Cote Winamax min"] = ((1 + edge) * 100 / df["Proba finale %"]).round(2)
    df["Value si cote = meilleure US %"] = ((df["Proba finale %"] / 100 * df["Meilleure US"] - 1) * 100).round(1)

    # ---- 1. shortlist
    st.markdown("**À regarder sur Winamax** : si la cote Winamax du joueur est ≥ « cote min », c'est du value selon l'outil")
    short = df.sort_values("Value si cote = meilleure US %", ascending=False, na_position="last").head(6)
    st.dataframe(
        short[["Joueur", "Équipe", "Proba finale %", "Cote Winamax min", "Meilleure US",
               "Value si cote = meilleure US %"]],
        hide_index=True,
    )
    st.caption("« Value si cote = meilleure US » suppose que Winamax offre la meilleure cote américaine. "
               f"Marge du marché supposée : {margin * 100:.0f} %. Poids du modèle : {w_model * 100:.0f} %.")

    # ---- 2. enregistrement des predictions
    out = pd.DataFrame({
        "date": day.isoformat(), "game_id": g["id"], "player_id": df["_id"], "joueur": df["Joueur"],
        "equipe": df["Équipe"], "p_model": df["Proba but %"], "p_mkt": df["Proba marché %"],
        "p_final": df["Proba finale %"], "best_us": df["Meilleure US"], "avg_us": df["Cote US moy."],
    })
    token, repo = gh_conf()
    if token and repo:
        if st.button("💾 Enregistrer les prédictions de ce match", key=f"save_{g['id']}"):
            try:
                gh_save(f"predictions/{day.isoformat()}_{g['id']}.csv", out.to_csv(index=False), token, repo)
                gh_load_predictions.clear()
                st.success("Enregistré. Le bilan se met à jour après la fin du match.")
            except Exception as e:
                st.error(f"Échec de l'enregistrement : {e}")
    else:
        st.download_button("⬇️ Télécharger les prédictions (CSV)", out.to_csv(index=False),
                           file_name=f"predictions_{day.isoformat()}_{g['id']}.csv", key=f"dl_{g['id']}")

    # ---- 3. detail + saisie optionnelle
    with st.expander("Détail par joueur et saisie d'une cote Winamax (optionnel)"):
        view = df[["Joue", "Joueur", "Équipe", "Proba but %", "Proba marché %", "Proba finale %",
                   "Cote juste", "Cote Winamax min", "Cote US moy.", "Meilleure US",
                   "Tirs/match", "% réussite", "Forme", "Buts attendus"]].copy()
        view["Cote Winamax"] = None
        edited = st.data_editor(
            view, hide_index=True, key=f"ed_{g['id']}",
            disabled=[c for c in view.columns if c not in ("Joue", "Cote Winamax")],
            column_config={"Cote Winamax": st.column_config.NumberColumn(min_value=1.01, step=0.05, format="%.2f")},
        )
        res = edited[(edited["Joue"]) & (edited["Cote Winamax"].fillna(0) > 1)].copy()
        if not res.empty:
            p = res["Proba finale %"] / 100
            res["Value %"] = ((p * res["Cote Winamax"] - 1) * 100).round(1)
            kelly = ((p * res["Cote Winamax"] - 1) / (res["Cote Winamax"] - 1)).clip(lower=0) * KELLY_FRACTION
            res["Mise suggérée (€)"] = (kelly * bankroll).round(2)
            st.dataframe(
                res.sort_values("Value %", ascending=False)[
                    ["Joueur", "Proba finale %", "Cote juste", "Cote Winamax", "Value %", "Mise suggérée (€)"]],
                hide_index=True,
            )
    st.caption("Vérifie toujours compos, blessures et gardien confirmé avant de jouer.")


def main():
    check_password()
    st.title("🏒 NHL – Buteurs value")
    st.caption("V2 – modèle + marché US. Outil d'aide à la décision, aucune garantie de gain.")

    tz = ZoneInfo("America/New_York")
    day = st.sidebar.date_input("Date NHL (heure US)", value=datetime.now(tz).date())
    bankroll = st.sidebar.number_input("Bankroll (€)", min_value=0.0, value=100.0, step=10.0)
    n = st.sidebar.slider("Joueurs analysés par équipe", 4, 14, 8)
    margin = st.sidebar.slider("Marge estimée du marché US (%)", 0, 20, 8) / 100
    w_model = st.sidebar.slider("Poids du modèle dans la proba finale (%)", 0, 100, 50) / 100
    edge = st.sidebar.slider("Avantage minimum exigé (%)", 0, 20, 5) / 100
    st.sidebar.markdown("**Convertisseur cote américaine**")
    am = st.sidebar.number_input("Cote US (ex. +150 ou -200)", value=150, step=5)
    if abs(am) >= 100:
        st.sidebar.write(f"= **{american_to_decimal(am):.2f}** en décimal (proba {100 / american_to_decimal(am):.1f} %)")
    if st.sidebar.button("Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()

    page = st.radio("Page", ["Matchs du jour", "Bilan"], horizontal=True, label_visibility="collapsed")
    if page == "Bilan":
        render_bilan()
        return

    season, prior = season_ids(day)
    try:
        games = get_games(day)
    except Exception as e:
        st.error(f"Impossible de lire le calendrier NHL : {e}")
        return
    if not games:
        st.info("Aucun match NHL à cette date.")
        return

    ga_map, league_pg = league_context()
    tabs = st.tabs([f"{g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}" for g in games])
    for tab, g in zip(tabs, games):
        with tab:
            render_game(g, season, prior, ga_map, league_pg, n, bankroll, margin, w_model, edge, day)


main()
