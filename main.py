from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
import multiprocessing as mp

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from models import TeamOptimizationResult, PlayersRequest, Player, SelectedPlayer, MVPOptimizationResult, MultiTeamOptimizationResult, MultiMVPOptimizationResult

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
        request: PlayersRequest
            players (List[Player]): Liste des joueurs disponibles
            cost (Integer): Coût maximum pour l'équipe
            team_size (Integer): Nombre de joueurs dans l'équipe
            minimum_stars (Integer): Nombre minimum de stars dans l'équipe

    Returns:
        TeamOptimizationResult: Composition optimale de l'équipe

    Raises:
        HTTPException: Si aucune solution optimale n'est trouvée
    """
    players = request.players
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []
    
    if players is None or cost is None or team_size is None or minimum_stars is None:
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )
    n = len(players)
    forced_indices = [i for i in range(n) if players[i].name in forced_players]

    prob = LpProblem("NBA_Team_Optimization", LpMaximize)

    x = [
        LpVariable(f"player_{i}_{players[i].name.replace(' ', '_')}", cat=LpBinary)
        for i in range(n)
    ]

    prob += lpSum([players[i].score * x[i] for i in range(n)]), "Total_Score"

    prob += lpSum([x[i] for i in range(n)]) == team_size, "Team_Size"

    prob += lpSum([players[i].cost * x[i] for i in range(n)]) <= cost, "Budget_Constraint"

    prob += lpSum([int(players[i].is_star) * x[i] for i in range(n)]) >= minimum_stars, "Minimum_Stars"

    for idx in forced_indices:
        prob += x[idx] == 1, f"Forced_Player_{idx}"
        
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
        request: PlayersRequest
            players (List[Player]): Liste des joueurs disponibles
            cost (Integer): Coût maximum pour l'équipe
            team_size (Integer): Nombre de joueurs dans l'équipe
            minimum_stars (Integer): Nombre minimum de stars dans l'équipe

    Returns:
        MVPOptimizationResult: Composition optimale avec joueur gratuit

    Raises:
        HTTPException: Si aucune solution optimale n'est trouvée
    """
    players = request.players
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []

    if players is None or cost is None or team_size is None or minimum_stars is None:
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )
    n = len(players)
    forced_indices = [i for i in range(n) if players[i].name in forced_players]

    score_max: float = max(p.score for p in players)

    prob = LpProblem("NBA_MVP_Optimise", LpMaximize)

    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
    y = [LpVariable(f"y_{i}", cat=LpBinary) for i in range(n)]

    s_gratuit = LpVariable("score_gratuit", lowBound=0, upBound=score_max)

    prob += lpSum([players[i].score * x[i] for i in range(n)]), "Score_Total"

    prob += lpSum([x[i] for i in range(n)]) == team_size, "Nb_Joueurs"

    prob += (
            lpSum([players[i].cost * x[i] for i in range(n)]) -
            lpSum([players[i].cost * y[i] for i in range(n)]) <= cost
    ), "Budget"

    prob += lpSum([int(players[i].is_star) * x[i] for i in range(n)]) >= minimum_stars, "Min_Etoiles"

    prob += lpSum([y[i] for i in range(n)]) == 1, "Un_Gratuit"

    for i in range(n):
        prob += y[i] <= x[i], f"Gratuit_Selec_{i}"

        prob += s_gratuit >= players[i].score * y[i], f"Score_Gratuit_{i}"

        prob += (
                players[i].score * x[i] <= s_gratuit + score_max * (1 - x[i])
        ), f"Max_Score_{i}"

    for idx in forced_indices:
        prob += x[idx] == 1, f"Forced_Player_{idx}"

    # n_threads = mp.cpu_count()
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))

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


@app.post("/best-comps", response_model=MultiTeamOptimizationResult)
async def find_top_compositions(
    request: PlayersRequest,
    top_n: int = 8
) -> MultiTeamOptimizationResult:
    """
    Trouve les N meilleures compositions d'équipe NBA
    """
    players = request.players        
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []
    
    if not all([players, cost is not None, team_size is not None, minimum_stars is not None]):
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )
    
    n = len(players)
    solutions = []
    excluded_combinations = []
    forced_indices = [i for i in range(n) if players[i].name in forced_players]

    player_scores = [players[i].score for i in range(n)]
    player_costs = [players[i].cost for i in range(n)]
    player_is_star = [int(players[i].is_star) for i in range(n)]
    
    for iteration in range(top_n):
        prob = LpProblem(f"NBA_{iteration}", LpMaximize)
        
        x = [LpVariable(f"p{i}_{iteration}", cat=LpBinary) for i in range(n)]
        
        prob += lpSum([player_scores[i] * x[i] for i in range(n)])
        
        prob += lpSum(x) == team_size
        prob += lpSum([player_costs[i] * x[i] for i in range(n)]) <= cost
        prob += lpSum([player_is_star[i] * x[i] for i in range(n)]) >= minimum_stars

        for idx in forced_indices:
            prob += x[idx] == 1
        
        for idx, excluded_idx in enumerate(excluded_combinations):
            prob += lpSum([x[i] for i in excluded_idx]) <= team_size - 1
        
        # n_threads = mp.cpu_count()
        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
        
        if LpStatus[prob.status] != 'Optimal':
            break
        
        selected_indices = [i for i in range(n) if x[i].varValue and x[i].varValue > 0.5]
        
        if not selected_indices:
            break
        
        selected_team = [players[i] for i in selected_indices]
        
        total_cost = sum(p.cost for p in selected_team)
        total_score = sum(p.score for p in selected_team)
        star_count = sum(1 for p in selected_team if p.is_star)
        
        solutions.append(TeamOptimizationResult(
            players=[p.name for p in selected_team],
            total_cost=total_cost,
            total_score=round(total_score, 2),
            star_count=star_count,
            details=[
                SelectedPlayer(name=p.name, cost=p.cost, score=p.score, is_star=p.is_star)
                for p in selected_team
            ],
            status="Optimal"
        ))
        
        excluded_combinations.append(selected_indices)
    
    if not solutions:
        raise HTTPException(status_code=400, detail="Aucune solution trouvée")
    
    return MultiTeamOptimizationResult(
        solutions=solutions,
        total_solutions_found=len(solutions)
    )

@app.post("/best-comps-mvp", response_model=MultiMVPOptimizationResult)
async def find_best_compositions_mvp(
    request: PlayersRequest,
    top_n: int = 8
) -> MultiMVPOptimizationResult:
    """
    Trouve les N meilleures compositions d'équipe NBA avec un joueur gratuit (MVP)
    Le joueur avec le coût le plus élevé parmi les sélectionnés est gratuit.
    
    Contraintes :
    - Exactement team_size joueurs
    - Budget maximum : cost (après déduction du joueur gratuit)
    - Minimum minimum_stars stars (All-Stars)
    - Un joueur est gratuit (le plus cher)
    - Le score de tous les joueurs doit être <= au score du joueur gratuit
    
    Args:
        request: PlayersRequest
        top_n: Nombre de solutions à retourner (défaut: 5)
    
    Returns:
        MultiMVPOptimizationResult: Les N meilleures compositions optimales avec joueur gratuit
    """
    players = request.players
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []
    
    if not all([players, cost is not None, team_size is not None, minimum_stars is not None]):
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )
    
    n = len(players)
    score_max = max(p.score for p in players)
    solutions = []
    excluded_combinations = []
    forced_indices = [i for i in range(n) if players[i].name in forced_players]
    
    for iteration in range(top_n):
        prob = LpProblem(f"NBA_MVP_Optimise_{iteration}", LpMaximize)
        
        x = [LpVariable(f"x_{i}_it{iteration}", cat=LpBinary) for i in range(n)]
        y = [LpVariable(f"y_{i}_it{iteration}", cat=LpBinary) for i in range(n)]
        s_gratuit = LpVariable(f"score_gratuit_it{iteration}", lowBound=0, upBound=score_max)
        
        prob += lpSum([players[i].score * x[i] for i in range(n)]), "Score_Total"
        
        prob += lpSum([x[i] for i in range(n)]) == team_size, "Nb_Joueurs"
        prob += (
            lpSum([players[i].cost * x[i] for i in range(n)]) -
            lpSum([players[i].cost * y[i] for i in range(n)]) <= cost
        ), "Budget"
        prob += lpSum([int(players[i].is_star) * x[i] for i in range(n)]) >= minimum_stars, "Min_Etoiles"
        prob += lpSum([y[i] for i in range(n)]) == 1, "Un_Gratuit"
        
        for i in range(n):
            prob += y[i] <= x[i], f"Gratuit_Selec_{i}"
            prob += s_gratuit >= players[i].score * y[i], f"Score_Gratuit_{i}"
            prob += (
                players[i].score * x[i] <= s_gratuit + score_max * (1 - x[i])
            ), f"Max_Score_{i}"

        for idx in forced_indices:
            prob += x[idx] == 1, f"Forced_Player_{idx}"
        
        for idx, excluded_indices in enumerate(excluded_combinations):
            prob += (
                lpSum([x[i] for i in excluded_indices]) <= team_size - 1,
                f"Exclude_Sol_it{iteration}_ex{idx}"
            )
        

        prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
        status = LpStatus[prob.status]
        
        if status != 'Optimal':
            break 

        selected_indices = [i for i in range(n) if x[i].varValue == 1]
        selected_team = [players[i] for i in selected_indices]
        free_player = None
        
        for i in range(n):
            if y[i].varValue == 1:
                free_player = players[i]
                break
        
        total_cost = sum(player.cost for player in selected_team)
        paid_cost = total_cost - (free_player.cost if free_player else 0)
        total_score = sum(player.score for player in selected_team)
        star_count = sum(1 for player in selected_team if player.is_star)
        
        solutions.append(MVPOptimizationResult(
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
        ))
        
        excluded_combinations.append(selected_indices)
    
    if len(solutions) == 0:
        raise HTTPException(
            status_code=400,
            detail="Aucune solution optimale trouvée. Vérifiez les contraintes."
        )
    
    return MultiMVPOptimizationResult(
        solutions=solutions,
        total_solutions_found=len(solutions)
    )