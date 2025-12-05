from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Player(BaseModel):
    name: str = Field(..., description="Nom du joueur")
    cost: int = Field(..., ge=0, description="Coût du joueur")
    score: float = Field(..., ge=0, description="Score/performance du joueur")
    is_star: bool = Field(..., description="Joueur de saison")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "LeBron James",
                "cost": 30,
                "score": 95.5,
                "is_star": True
            }
        }


class PlayersRequest(BaseModel):
    players: List[Player] = Field(..., min_length=5, description="Liste des joueurs disponibles")
    cost: int = Field(..., ge=0, description="Coût maximum pour l'équipe")
    team_size: int = Field(..., ge=1, description="Nombre de joueurs dans l'équipe")
    minimum_stars: int = Field(..., ge=0, description="Nombre minimum de stars dans l'équipe")
    forced_players: Optional[List[str]] = Field(default=None, description="Noms des joueurs obligatoires")

    @field_validator('players')
    @classmethod
    def validate_players(cls, v: List[Player]) -> List[Player]:
        if len(v) < 5:
            raise ValueError("Au moins 5 joueurs sont nécessaires pour former une équipe")
        return v


class SelectedPlayer(BaseModel):
    name: str
    cost: int
    score: float
    is_star: bool

class TeamOptimizationResult(BaseModel):
    players: List[str] = Field(..., description="Noms des joueurs sélectionnés")
    total_cost: int = Field(..., description="Coût total de l'équipe")
    total_score: float = Field(..., description="Score total de l'équipe")
    star_count: int = Field(..., description="Nombre de joueur de saison dans l'équipe")
    details: List[SelectedPlayer] = Field(..., description="Détails complets des joueurs sélectionnés")
    status: str = Field(..., description="Statut de l'optimisation")

class MultiTeamOptimizationResult(BaseModel):
    solutions: List[TeamOptimizationResult] = Field(..., description="Top N meilleures compositions")
    total_solutions_found: int = Field(..., description="Nombre total de solutions trouvées")

class MVPOptimizationResult(BaseModel):
    players: List[str] = Field(..., description="Noms des joueurs sélectionnés")
    free_player: Optional[str] = Field(None, description="Nom du joueur gratuit (plus cher)")
    total_cost: int = Field(..., description="Coût total réel de l'équipe")
    paid_cost: int = Field(..., description="Coût payé (sans le joueur gratuit)")
    total_score: float = Field(..., description="Score total de l'équipe")
    star_count: int = Field(..., description="Nombre de stars dans l'équipe")
    details: List[SelectedPlayer] = Field(..., description="Détails complets des joueurs sélectionnés")
    status: str = Field(..., description="Statut de l'optimisation")

    class Config:
        json_schema_extra = {
            "example": {
                "players": ["LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo", "Nikola Jokic"],
                "total_cost": 120,
                "total_score": 475.5,
                "star_count": 5,
                "details": [],
                "status": "Optimal"
            }
        }

class MultiMVPOptimizationResult(BaseModel):
    solutions: List[MVPOptimizationResult] = Field(..., description="Top N meilleures compositions MVP")
    total_solutions_found: int = Field(..., description="Nombre total de solutions trouvées")