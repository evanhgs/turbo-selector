from typing import List, Optional, Dict, Any, Union, Set
from fastapi import FastAPI, HTTPException
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD
import multiprocessing as mp

from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from models import TeamOptimizationResult, PlayersRequest, Player, SelectedPlayer, MVPOptimizationResult, \
    MultiTeamOptimizationResult, MultiMVPOptimizationResult, TopNResult

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


@app.post("/best-comp", response_model=Union[TeamOptimizationResult, MVPOptimizationResult])
async def find_best_composition(
        request: PlayersRequest
) -> Union[TeamOptimizationResult, MVPOptimizationResult]:
    """
    Trouve la meilleure composition d'équipe NBA en maximisant le score

    Mode standard (is_mvp=False):
    - Contraintes classiques : budget, taille d'équipe, stars minimum

    Mode MVP (is_mvp=True):
    - Le joueur avec le coût le plus élevé est gratuit
    - Le score des autres joueurs doit être <= au score du joueur gratuit

    OPTIMISATION : Si des joueurs sont forcés, le problème est réduit aux places restantes
    uniquement, ce qui accélère considérablement la résolution.

    Args:
        request: PlayersRequest contenant tous les paramètres nécessaires

    Returns:
        TeamOptimizationResult ou MVPOptimizationResult selon le mode

    Raises:
        HTTPException: Si les paramètres sont manquants ou aucune solution trouvée
    """
    players = request.players
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []
    is_mvp = request.is_mvp or False

    # Validation des paramètres
    if players is None or cost is None or team_size is None or minimum_stars is None:
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )

    n = len(players)

    # ============================================================================
    # OPTIMISATION : Prétraitement des joueurs forcés
    # ============================================================================
    forced_team: List[Player] = []
    available_players: List[Player] = []
    available_indices_mapping: List[int] = []  # Mapping vers les indices originaux

    for i, player in enumerate(players):
        if player.name in forced_players:
            forced_team.append(player)
        else:
            available_players.append(player)
            available_indices_mapping.append(i)

    # Validation des joueurs forcés
    if len(forced_team) != len(forced_players):
        missing = set(forced_players) - {p.name for p in forced_team}
        raise HTTPException(
            status_code=400,
            detail=f"Joueurs forcés introuvables : {missing}"
        )

    if len(forced_team) > team_size:
        raise HTTPException(
            status_code=400,
            detail=f"Trop de joueurs forcés ({len(forced_team)}) pour une équipe de {team_size}"
        )

    # Calcul des contraintes restantes après joueurs forcés
    forced_cost = sum(p.cost for p in forced_team)
    forced_stars = sum(1 for p in forced_team if p.is_star)
    forced_score = sum(p.score for p in forced_team)

    remaining_budget = cost - forced_cost
    remaining_stars_needed = max(0, minimum_stars - forced_stars)
    remaining_slots = team_size - len(forced_team)

    # Validation de faisabilité
    if remaining_budget < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Les joueurs forcés dépassent le budget. "
                   f"Coût forcé: {forced_cost}, Budget: {cost}"
        )

    if remaining_stars_needed > remaining_slots:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible d'atteindre {minimum_stars} stars. "
                   f"Forcés: {forced_stars}, Places restantes: {remaining_slots}"
        )

    # ============================================================================
    # CAS PARTICULIER : Tous les joueurs sont forcés
    # ============================================================================
    if remaining_slots == 0:
        return _build_result(
            selected_team=forced_team,
            free_player=None,
            is_mvp=is_mvp,
            status="Optimal"
        )

    # ============================================================================
    # OPTIMISATION MODE MVP : Contrainte sur le score du joueur gratuit
    # ============================================================================
    max_forced_score = max((p.score for p in forced_team), default=0) if forced_team else 0

    if is_mvp and forced_team:
        # En mode MVP, si un joueur forcé existe, il pourrait être le joueur gratuit
        # On doit donc filtrer les joueurs disponibles selon leur score
        # MAIS on ne peut pas savoir à l'avance qui sera gratuit, donc on garde tous
        # et on laisse l'optimisation gérer cette contrainte
        pass

    # ============================================================================
    # RÉSOLUTION DU SOUS-PROBLÈME avec les joueurs disponibles uniquement
    # ============================================================================
    n_available = len(available_players)

    prob = LpProblem(
        f"NBA_Optimisation_{'MVP' if is_mvp else 'Standard'}_Reduced",
        LpMaximize
    )

    # Variables de décision pour les joueurs DISPONIBLES uniquement
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n_available)]

    # Variables spécifiques au mode MVP
    if is_mvp:
        # Le score max inclut maintenant les joueurs forcés ET disponibles
        score_max = max(
            max(p.score for p in available_players) if available_players else 0,
            max_forced_score
        )
        # y peut cibler les joueurs disponibles OU forcés
        # Pour simplifier, on créé des variables pour disponibles + forcés
        y_available = [LpVariable(f"y_avail_{i}", cat=LpBinary) for i in range(n_available)]
        y_forced = [LpVariable(f"y_forced_{i}", cat=LpBinary) for i in range(len(forced_team))]
        s_gratuit = LpVariable("score_gratuit", lowBound=0, upBound=score_max)

    # ============================================================================
    # FONCTION OBJECTIF : Maximiser le score des joueurs disponibles
    # (le score des forcés est déjà acquis)
    # ============================================================================
    prob += lpSum([available_players[i].score * x[i] for i in range(n_available)]), "Total_Score"

    # ============================================================================
    # CONTRAINTES
    # ============================================================================

    # 1. Nombre de joueurs à sélectionner (places restantes)
    prob += lpSum([x[i] for i in range(n_available)]) == remaining_slots, "Remaining_Slots"

    # 2. Contrainte budgétaire
    if is_mvp:
        # En mode MVP, on doit considérer qu'un joueur sera gratuit
        # Budget disponible : remaining_budget + coût du joueur gratuit
        prob += (
                lpSum([available_players[i].cost * x[i] for i in range(n_available)])
                - lpSum([available_players[i].cost * y_available[i] for i in range(n_available)])
                - lpSum([forced_team[i].cost * y_forced[i] for i in range(len(forced_team))])
                <= remaining_budget
        ), "Budget_Constraint"

        # Un seul joueur gratuit (parmi disponibles OU forcés)
        prob += (
                lpSum([y_available[i] for i in range(n_available)])
                + lpSum([y_forced[i] for i in range(len(forced_team))])
                == 1
        ), "One_Free_Player"
    else:
        prob += (
            lpSum([available_players[i].cost * x[i] for i in range(n_available)]) <= remaining_budget,
            "Budget_Constraint"
        )

    # 3. Contrainte sur les stars (places restantes nécessaires)
    prob += (
        lpSum([int(available_players[i].is_star) * x[i] for i in range(n_available)]) >= remaining_stars_needed,
        "Remaining_Stars"
    )

    # ============================================================================
    # CONTRAINTES SPÉCIFIQUES MODE MVP
    # ============================================================================
    if is_mvp:
        # Le joueur gratuit doit être sélectionné (pour les disponibles)
        for i in range(n_available):
            prob += y_available[i] <= x[i], f"Free_Selected_Avail_{i}"
            prob += s_gratuit >= available_players[i].score * y_available[i], f"Free_Score_Avail_{i}"
            prob += (
                available_players[i].score * x[i] <= s_gratuit + score_max * (1 - x[i]),
                f"Max_Score_Avail_{i}"
            )

        # Pour les joueurs forcés (toujours sélectionnés)
        for i in range(len(forced_team)):
            prob += s_gratuit >= forced_team[i].score * y_forced[i], f"Free_Score_Forced_{i}"
            # Les joueurs forcés doivent aussi respecter la contrainte de score
            prob += (
                forced_team[i].score <= s_gratuit + score_max * (1 - y_forced[i]),
                f"Max_Score_Forced_{i}"
            )

    # ============================================================================
    # RÉSOLUTION
    # ============================================================================
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    status = LpStatus[prob.status]

    if status != 'Optimal':
        raise HTTPException(
            status_code=400,
            detail=f"Aucune solution optimale trouvée. Statut: {status}. "
                   f"Vérifiez les contraintes (budget restant: {remaining_budget}, "
                   f"places restantes: {remaining_slots}, stars nécessaires: {remaining_stars_needed})"
        )

    # ============================================================================
    # CONSTRUCTION DE L'ÉQUIPE FINALE
    # ============================================================================
    selected_available: List[Player] = []
    free_player: Optional[Player] = None

    for i in range(n_available):
        if x[i].varValue == 1:
            selected_available.append(available_players[i])
            if is_mvp and y_available[i].varValue == 1:
                free_player = available_players[i]

    # Vérifier si un joueur forcé est le joueur gratuit
    if is_mvp and free_player is None:
        for i in range(len(forced_team)):
            if y_forced[i].varValue == 1:
                free_player = forced_team[i]
                break

    # Équipe finale = forcés + sélectionnés
    final_team = forced_team + selected_available

    return _build_result(
        selected_team=final_team,
        free_player=free_player,
        is_mvp=is_mvp,
        status=status
    )

@app.post("/best-comp/top-n", response_model=TopNResult)
async def find_top_n_compositions(
        request: PlayersRequest,
        n: int = 20
) -> TopNResult:
    """
    Trouve les N meilleures compositions d'équipe NBA par ordre décroissant de score

    OPTIMISATIONS :
    - Utilise la réduction du problème pour les joueurs forcés
    - Méthode itérative avec exclusion pour trouver les solutions suivantes
    - Stop anticipé si aucune nouvelle solution n'est trouvée

    Args:
        request: PlayersRequest contenant tous les paramètres
        n: Nombre de compositions à retourner (défaut: 8)

    Returns:
        TopNResult contenant les N meilleures compositions triées par score

    Raises:
        HTTPException: Si les paramètres sont invalides ou aucune solution trouvée
    """
    if n < 1 or n > 100:
        raise HTTPException(
            status_code=400,
            detail="Le paramètre n doit être entre 1 et 100"
        )

    players = request.players
    cost = request.cost
    team_size = request.team_size
    minimum_stars = request.minimum_stars
    forced_players = request.forced_players or []
    is_mvp = request.is_mvp or False

    # Validation des paramètres
    if players is None or cost is None or team_size is None or minimum_stars is None:
        raise HTTPException(
            status_code=400,
            detail="Les paramètres players, cost, team_size et minimum_stars sont requis"
        )

    results: List[Union[TeamOptimizationResult, MVPOptimizationResult]] = []
    excluded_combinations: List[Set[str]] = []

    # ============================================================================
    # PRÉTRAITEMENT : Séparation forcés / disponibles (comme avant)
    # ============================================================================
    forced_team, available_players, validation_result = _preprocess_forced_players(
        players, forced_players, team_size, cost, minimum_stars
    )

    if validation_result:  # Erreur de validation
        raise HTTPException(status_code=400, detail=validation_result)

    remaining_budget = cost - sum(p.cost for p in forced_team)
    remaining_stars_needed = max(0, minimum_stars - sum(1 for p in forced_team if p.is_star))
    remaining_slots = team_size - len(forced_team)

    # Cas particulier : tous les joueurs sont forcés
    if remaining_slots == 0:
        result = _build_result(
            selected_team=forced_team,
            free_player=None,
            is_mvp=is_mvp,
            status="Optimal"
        )
        return TopNResult(
            results=[result],
            total_found=1,
            requested=n,
        )

    # ============================================================================
    # BOUCLE PRINCIPALE : Trouver les N meilleures solutions
    # ============================================================================
    for iteration in range(n):
        try:
            # Résoudre le problème en excluant les solutions déjà trouvées
            result = _solve_optimization_with_exclusions(
                available_players=available_players,
                forced_team=forced_team,
                remaining_slots=remaining_slots,
                remaining_budget=remaining_budget,
                remaining_stars_needed=remaining_stars_needed,
                excluded_combinations=excluded_combinations,
                is_mvp=is_mvp,
                iteration=iteration
            )

            if result is None:
                break

            results.append(result)

            # Ajouter cette combinaison à la liste d'exclusion
            current_combination = set(result.players)
            excluded_combinations.append(current_combination)

        except Exception as e:
            print(f"Arrêt à l'itération {iteration}: {e}")
            break

    if not results:
        raise HTTPException(
            status_code=400,
            detail="Aucune composition valide trouvée avec les contraintes données"
        )

    return TopNResult(
        results=results,
        total_found=len(results),
        requested=n,
    )

# ============================================================================
# FONCTION DE PRÉTRAITEMENT
# ============================================================================
def _preprocess_forced_players(
        players: List[Player],
        forced_players: List[str],
        team_size: int,
        cost: int,
        minimum_stars: int
) -> tuple[List[Player], List[Player], Optional[str]]:
    """
    Sépare les joueurs forcés des joueurs disponibles et valide les contraintes

    Returns:
        (forced_team, available_players, error_message)
        error_message est None si tout est valide
    """
    forced_team: List[Player] = []
    available_players: List[Player] = []

    for player in players:
        if player.name in forced_players:
            forced_team.append(player)
        else:
            available_players.append(player)

    # Validations
    if len(forced_team) != len(forced_players):
        missing = set(forced_players) - {p.name for p in forced_team}
        return forced_team, available_players, f"Joueurs forcés introuvables : {missing}"

    if len(forced_team) > team_size:
        return forced_team, available_players, \
            f"Trop de joueurs forcés ({len(forced_team)}) pour une équipe de {team_size}"

    forced_cost = sum(p.cost for p in forced_team)
    if forced_cost > cost:
        return forced_team, available_players, \
            f"Les joueurs forcés dépassent le budget. Coût: {forced_cost}, Budget: {cost}"

    forced_stars = sum(1 for p in forced_team if p.is_star)
    remaining_slots = team_size - len(forced_team)
    remaining_stars_needed = max(0, minimum_stars - forced_stars)

    if remaining_stars_needed > remaining_slots:
        return forced_team, available_players, \
            f"Impossible d'atteindre {minimum_stars} stars. Forcés: {forced_stars}, Places: {remaining_slots}"

    return forced_team, available_players, None


# ============================================================================
# FONCTION DE RÉSOLUTION AVEC EXCLUSIONS
# ============================================================================
def _solve_optimization_with_exclusions(
        available_players: List[Player],
        forced_team: List[Player],
        remaining_slots: int,
        remaining_budget: int,
        remaining_stars_needed: int,
        excluded_combinations: List[Set[str]],
        is_mvp: bool,
        iteration: int
) -> Optional[Union[TeamOptimizationResult, MVPOptimizationResult]]:
    """
    Résout le problème d'optimisation en excluant les combinaisons déjà trouvées

    Returns:
        Le résultat optimal ou None si aucune solution n'existe
    """
    n_available = len(available_players)

    if n_available == 0 and remaining_slots > 0:
        return None

    prob = LpProblem(
        f"NBA_Top_{iteration}_{'MVP' if is_mvp else 'Standard'}",
        LpMaximize
    )

    # Variables de décision
    x = [LpVariable(f"x_{i}_iter{iteration}", cat=LpBinary) for i in range(n_available)]

    # Variables MVP
    if is_mvp:
        max_forced_score = max((p.score for p in forced_team), default=0) if forced_team else 0
        score_max = max(
            max(p.score for p in available_players) if available_players else 0,
            max_forced_score
        )
        y_available = [LpVariable(f"y_avail_{i}_iter{iteration}", cat=LpBinary) for i in range(n_available)]
        y_forced = [LpVariable(f"y_forced_{i}_iter{iteration}", cat=LpBinary) for i in range(len(forced_team))]
        s_gratuit = LpVariable(f"score_gratuit_iter{iteration}", lowBound=0, upBound=score_max)

    # Fonction objectif
    prob += lpSum([available_players[i].score * x[i] for i in range(n_available)]), "Total_Score"

    # Contraintes de base
    prob += lpSum([x[i] for i in range(n_available)]) == remaining_slots, "Remaining_Slots"

    if is_mvp:
        prob += (
                lpSum([available_players[i].cost * x[i] for i in range(n_available)])
                - lpSum([available_players[i].cost * y_available[i] for i in range(n_available)])
                - lpSum([forced_team[i].cost * y_forced[i] for i in range(len(forced_team))])
                <= remaining_budget
        ), "Budget_Constraint"
        prob += (
                lpSum([y_available[i] for i in range(n_available)])
                + lpSum([y_forced[i] for i in range(len(forced_team))])
                == 1
        ), "One_Free_Player"
    else:
        prob += (
            lpSum([available_players[i].cost * x[i] for i in range(n_available)]) <= remaining_budget,
            "Budget_Constraint"
        )

    prob += (
        lpSum([int(available_players[i].is_star) * x[i] for i in range(n_available)]) >= remaining_stars_needed,
        "Remaining_Stars"
    )

    # Contraintes MVP
    if is_mvp:
        for i in range(n_available):
            prob += y_available[i] <= x[i], f"Free_Selected_Avail_{i}"
            prob += s_gratuit >= available_players[i].score * y_available[i], f"Free_Score_Avail_{i}"
            prob += (
                available_players[i].score * x[i] <= s_gratuit + score_max * (1 - x[i]),
                f"Max_Score_Avail_{i}"
            )

        for i in range(len(forced_team)):
            prob += s_gratuit >= forced_team[i].score * y_forced[i], f"Free_Score_Forced_{i}"
            prob += (
                forced_team[i].score <= s_gratuit + score_max * (1 - y_forced[i]),
                f"Max_Score_Forced_{i}"
            )

    # ============================================================================
    # CONTRAINTES D'EXCLUSION : Empêcher les combinaisons déjà trouvées
    # ============================================================================
    for idx, excluded_combo in enumerate(excluded_combinations):
        # Une combinaison est différente si au moins un joueur diffère
        # On exclut les forcés de la comparaison (ils sont toujours présents)
        excluded_available = [p for p in excluded_combo if p not in {fp.name for fp in forced_team}]

        if not excluded_available:
            continue

        # Trouver les indices dans available_players
        excluded_indices = [
            i for i, p in enumerate(available_players)
            if p.name in excluded_available
        ]

        # Cette combinaison ne doit pas être exactement reproduite
        # Au moins un des joueurs précédemment sélectionnés ne doit PAS être sélectionné
        prob += (
            lpSum([x[i] for i in excluded_indices]) <= len(excluded_indices) - 1,
            f"Exclude_Combination_{idx}"
        )

    # Résolution
    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=15))
    status = LpStatus[prob.status]

    if status != 'Optimal':
        return None

    # Construction du résultat
    selected_available: List[Player] = []
    free_player: Optional[Player] = None

    for i in range(n_available):
        if x[i].varValue == 1:
            selected_available.append(available_players[i])
            if is_mvp and y_available[i].varValue == 1:
                free_player = available_players[i]

    if is_mvp and free_player is None:
        for i in range(len(forced_team)):
            if y_forced[i].varValue == 1:
                free_player = forced_team[i]
                break

    final_team = forced_team + selected_available

    return _build_result(
        selected_team=final_team,
        free_player=free_player,
        is_mvp=is_mvp,
        status=status
    )




# ============================================================================
# FONCTION UTILITAIRE pour construire le résultat
# ============================================================================
def _build_result(
        selected_team: List[Player],
        free_player: Optional[Player],
        is_mvp: bool,
        status: str
) -> Union[TeamOptimizationResult, MVPOptimizationResult]:
    """
    Construit le résultat final selon le mode (standard ou MVP)

    Args:
        selected_team: Liste des joueurs sélectionnés
        free_player: Joueur gratuit (None si mode standard)
        is_mvp: Mode MVP activé ou non
        status: Statut de la résolution

    Returns:
        TeamOptimizationResult ou MVPOptimizationResult
    """
    total_cost = sum(player.cost for player in selected_team)
    total_score = sum(player.score for player in selected_team)
    star_count = sum(1 for player in selected_team if player.is_star)

    details = [
        SelectedPlayer(
            name=player.name,
            cost=player.cost,
            score=player.score,
            is_star=player.is_star
        )
        for player in selected_team
    ]

    if is_mvp:
        paid_cost = total_cost - (free_player.cost if free_player else 0)
        return MVPOptimizationResult(
            players=[player.name for player in selected_team],
            free_player=free_player.name if free_player else None,
            total_cost=total_cost,
            paid_cost=paid_cost,
            total_score=round(total_score, 2),
            star_count=star_count,
            details=details,
            status=status
        )
    else:
        return TeamOptimizationResult(
            players=[player.name for player in selected_team],
            total_cost=total_cost,
            total_score=round(total_score, 2),
            star_count=star_count,
            details=details,
            status=status
        )

