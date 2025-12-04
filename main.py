from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
import multiprocessing as mp

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from models import TeamOptimizationResult, PlayersRequest, Player, SelectedPlayer, MVPOptimizationResult

app = FastAPI(
    title="NBA Team Optimizer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def root():
    return {
        "message": "NBA Team Optimizer API",
        "version": "1.0.0",
        "endpoints": {
            "health check": "/health",
            "docs url": "/docs",
            "redoc url": "/redoc",
            "open api": "/openapi.json",
            "standard": "/best-comp",
            "mvp": "/best-comp-mvp"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "nba-optimizer"}


@app.post("/best-comp", response_model=TeamOptimizationResult)
async def find_best_composition(request: PlayersRequest) -> TeamOptimizationResult:
    """
    Résultat vérifié par l'algo: prog dynamique et parcours en profondeur
    Trouve la meilleure composition d'équipe NBA en maximisant le score

    Contraintes :
    - Exactement 5 joueurs
    - Budget maximum : 120
    - Minimum 4 stars (All-Stars)

    Args:
        request: Liste des joueurs disponibles

    Returns:
        TeamOptimizationResult: Composition optimale de l'équipe

    Raises:
        HTTPException: Si aucune solution optimale n'est trouvée
    """
    players = request.players
    n = len(players)

    prob = LpProblem("NBA_Team_Optimization", LpMaximize)

    x = [
        LpVariable(f"player_{i}_{players[i].name.replace(' ', '_')}", cat=LpBinary)
        for i in range(n)
    ]

    prob += lpSum([players[i].score * x[i] for i in range(n)]), "Total_Score"

    prob += lpSum([x[i] for i in range(n)]) == 5, "Team_Size"

    prob += lpSum([players[i].cost * x[i] for i in range(n)]) <= 120, "Budget_Constraint"

    prob += lpSum([int(players[i].is_star) * x[i] for i in range(n)]) >= 4, "Minimum_Stars"

    prob.solve(PULP_CBC_CMD(msg=0))

    status = LpStatus[prob.status]
    if status != 'Optimal':
        raise HTTPException(
            status_code=400,
            detail=f"Aucune solution optimale trouvée. Statut: {status}. "
                   f"Vérifiez les contraintes (budget, nombre de stars, etc.)"
        )

    selected_team: List[Player] = [
        players[i] for i in range(n) if x[i].varValue == 1
    ]

    total_cost = sum(player.cost for player in selected_team)
    total_score = sum(player.score for player in selected_team)
    star_count = sum(1 for player in selected_team if player.is_star)

    return TeamOptimizationResult(
        players=[player.name for player in selected_team],
        total_cost=total_cost,
        total_score=round(total_score, 2),
        star_count=star_count,
        details=[
            SelectedPlayer(
                name=player.name,
                cost=player.cost,
                score=player.score,
                is_star=player.is_star
            )
            for player in selected_team
        ],
        status=status
    )


@app.post("/best-comp-mvp", response_model=MVPOptimizationResult)
async def find_best_composition_mvp(request: PlayersRequest) -> MVPOptimizationResult:
    """
    Trouve la meilleure composition d'équipe NBA avec un joueur gratuit (MVP)

    Le joueur avec le coût le plus élevé parmi les sélectionnés est gratuit.

    Contraintes :
    - Exactement 5 joueurs
    - Budget maximum : 120 (après déduction du joueur gratuit)
    - Minimum 4 stars (All-Stars)
    - Un joueur est gratuit (le plus cher)
    - Le score de tous les joueurs doit être <= au score du joueur gratuit

    Args:
        request: Liste des joueurs disponibles

    Returns:
        MVPOptimizationResult: Composition optimale avec joueur gratuit

    Raises:
        HTTPException: Si aucune solution optimale n'est trouvée
    """
    players = request.players
    n = len(players)

    score_max: float = max(p.score for p in players)

    prob = LpProblem("NBA_MVP_Optimise", LpMaximize)

    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
    y = [LpVariable(f"y_{i}", cat=LpBinary) for i in range(n)]

    s_gratuit = LpVariable("score_gratuit", lowBound=0, upBound=score_max)

    prob += lpSum([players[i].score * x[i] for i in range(n)]), "Score_Total"

    prob += lpSum([x[i] for i in range(n)]) == 5, "Nb_Joueurs"
    prob += (
            lpSum([players[i].cost * x[i] for i in range(n)]) -
            lpSum([players[i].cost * y[i] for i in range(n)]) <= 120
    ), "Budget"
    prob += lpSum([int(players[i].is_star) * x[i] for i in range(n)]) >= 4, "Min_Etoiles"
    prob += lpSum([y[i] for i in range(n)]) == 1, "Un_Gratuit"

    for i in range(n):
        prob += y[i] <= x[i], f"Gratuit_Selec_{i}"

        prob += s_gratuit >= players[i].score * y[i], f"Score_Gratuit_{i}"

        prob += (
                players[i].score * x[i] <= s_gratuit + score_max * (1 - x[i])
        ), f"Max_Score_{i}"

    n_threads = mp.cpu_count()
    prob.solve(PULP_CBC_CMD(msg=0, threads=n_threads, timeLimit=30))

    status = LpStatus[prob.status]
    if status != 'Optimal':
        raise HTTPException(
            status_code=400,
            detail=f"Aucune solution optimale trouvée. Statut: {status}. "
                   f"Vérifiez les contraintes (budget, nombre de stars, etc.)"
        )

    selected_team: List[Player] = []
    free_player: Optional[Player] = None

    for i in range(n):
        if x[i].varValue == 1:
            selected_team.append(players[i])
            if y[i].varValue == 1:
                free_player = players[i]

    total_cost = sum(player.cost for player in selected_team)
    paid_cost = total_cost - (free_player.cost if free_player else 0)
    total_score = sum(player.score for player in selected_team)
    star_count = sum(1 for player in selected_team if player.is_star)

    return MVPOptimizationResult(
        players=[player.name for player in selected_team],
        free_player=free_player.name if free_player else None,
        total_cost=total_cost,
        paid_cost=paid_cost,
        total_score=round(total_score, 2),
        star_count=star_count,
        details=[
            SelectedPlayer(
                name=player.name,
                cost=player.cost,
                score=player.score,
                is_star=player.is_star
            )
            for player in selected_team
        ],
        status=status
    )
