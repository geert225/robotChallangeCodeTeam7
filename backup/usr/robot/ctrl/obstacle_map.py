#!/usr/bin/env python3
"""
obstacle_map.py  —  Obstakelskaart op basis van ultrasoon + odometrie

Bouwt een schatting van obstakelposities in het wereldframe op.
Gebruikt door fullBrain.py voor obstakelontwijking tijdens HOME-navigatie.

Geen directe I2C of SHM-toegang: werkt puur op getallen die fullBrain
aanlevert. Dit maakt de module makkelijk te testen en herbruikbaar.

Werking:
  1. fullBrain roept add_reading() aan vanuit obstacle_map_loop()
     telkens als de ultrasoon een nieuwe meting heeft.
  2. add_reading() berekent de geschatte obstakelposition in het
     wereldframe en slaat hem op (nabije duplicaten worden samengevoegd).
  3. Tijdens HOME-navigatie roept home_loop() apply_repulsion() aan.
     Dit past de gewenste rijsnelheid aan met een afstotingsvector
     van alle nabije opgeslagen obstakels.

Coördinatenstelsel — zelfde als odometry.py / fullBrain.py:
  x vooruit (rijrichting bij start), y links, theta=0 bij start.
  Afstanden in meters, hoeken in radialen.
"""

import math
from typing import List, Tuple

# ─── Opname-parameters ───────────────────────────────────────────────────────
MAX_OBSTACLES     = 120    # maximaal aantal opgeslagen obstakels (FIFO bij overschrijding)
MERGE_RADIUS      = 0.20   # m — nieuwe meting binnen deze straal → samenvoegen
MAX_RECORD_DIST_M = 0.30   # m — alleen opslaan als sensor < 30 cm meet (dichtbij obstakel)
MIN_RECORD_DIST_M = 0.05   # m — minimale afstand (ruis/zelf-detectie filter)
HOME_EXCLUSION_R  = 0.30   # m — obstakels in deze straal rondom thuis (0,0) worden NIET opgeslagen
                           #     (voorkomt dat de home-zone als obstakel wordt vastgelegd)

# ─── Afstotings-parameters ───────────────────────────────────────────────────
INFLUENCE_RADIUS  = 0.70   # m — obstakels buiten deze straal hebben geen invloed
K_REPULSE         = 0.45   # versterkingsfactor afstoting (m/s / m)
MAX_REPULSE_SPD   = 0.55   # m/s — maximale afstotingssnelheid (clamp)
REPULSE_MIN_DIST  = 0.12   # m — ondersteuningsafstand (voorkomt deling door nul)


class ObstacleMap:
    """
    Geheugen voor geschatte obstakelposities in het wereldframe.

    Gebruik (vanuit fullBrain.py):

        om = ObstacleMap()

        # In obstacle_map_loop() (5 Hz):
        om.add_reading(px, py, theta, d_front_cm)

        # In home_loop() na berekening gewenste robotsnelheid:
        vx_robot, vy_robot = om.apply_repulsion(px, py, theta, vx_robot, vy_robot)

        # Kaart resetten (optioneel, via WS-commando):
        om.clear()

        # Alle obstakels opvragen (voor visualisatie / debug):
        obstacles = om.get_obstacles()   # List[Tuple[float, float]]
    """

    def __init__(self):
        self._obstacles: List[Tuple[float, float]] = []

    # ─── Opname ──────────────────────────────────────────────────────────────
    def add_reading(self, px: float, py: float, theta: float,
                    d_cm: int, direction_offset: float = 0.0) -> None:
        """
        Voegt een ultrasoon-meting toe aan de kaart.

        Parameters
        ----------
        px, py           : huidige robotpositie in het wereldframe [m]
        theta            : huidige robotkop [rad]
        d_cm             : gemeten afstand [cm].  0 = geen geldig signaal.
        direction_offset : hoekoffset t.o.v. theta [rad].
                           0.0  = voorsensor (rijrichting),
                           math.pi = achtersensor.
        """
        if d_cm <= 0:
            return

        d_m = d_cm / 100.0

        # Buiten opname-venster → negeren
        if not (MIN_RECORD_DIST_M <= d_m <= MAX_RECORD_DIST_M):
            return

        # Berekening obstakelposition in wereldframe
        angle = theta + direction_offset
        ox = px + d_m * math.cos(angle)
        oy = py + d_m * math.sin(angle)

        # Sla geen obstakels op in de home-zone (rondom oorsprong)
        if math.sqrt(ox * ox + oy * oy) < HOME_EXCLUSION_R:
            return

        # Samenvoegen met bestaand obstakel als het dichtbij genoeg is
        for i, (ex, ey) in enumerate(self._obstacles):
            dx, dy = ox - ex, oy - ey
            if dx * dx + dy * dy < MERGE_RADIUS ** 2:
                # Gewogen gemiddelde: bestaand punt weegt zwaarder (stabielere schatting)
                self._obstacles[i] = (ex * 0.80 + ox * 0.20,
                                      ey * 0.80 + oy * 0.20)
                return

        # Nieuw obstakel toevoegen; FIFO als de lijst vol is
        if len(self._obstacles) >= MAX_OBSTACLES:
            self._obstacles.pop(0)
        self._obstacles.append((ox, oy))

    # ─── Afstoting ───────────────────────────────────────────────────────────
    def apply_repulsion(self, px: float, py: float, theta: float,
                        vx_robot: float, vy_robot: float) -> Tuple[float, float]:
        """
        Past de gewenste robotsnelheid aan met een afstotingsvector van
        alle nabije opgeslagen obstakels.

        Parameters
        ----------
        px, py     : huidige robotpositie in het wereldframe [m]
        theta      : huidige robotkop [rad]
        vx_robot   : gewenste rijsnelheid vooruit in robotframe [m/s]
        vy_robot   : gewenste rijsnelheid zijwaarts in robotframe [m/s]

        Geeft
        -----
        (vx_robot_gecorr, vy_robot_gecorr)  —  gecorrigeerde snelheden
        """
        if not self._obstacles:
            return vx_robot, vy_robot

        rep_x = 0.0   # afstotingsvector in wereldframe
        rep_y = 0.0

        for ox, oy in self._obstacles:
            dx = px - ox
            dy = py - oy
            d2 = dx * dx + dy * dy
            d  = math.sqrt(d2)
            if d >= INFLUENCE_RADIUS or d < 0.001:
                continue

            # Afstotingskracht volgens inverse potential field
            # F = K * (1/d  -  1/R_inf) , richting weg van obstakel
            d_eff = max(d, REPULSE_MIN_DIST)
            mag   = K_REPULSE * (1.0 / d_eff - 1.0 / INFLUENCE_RADIUS)
            rep_x += mag * dx / d
            rep_y += mag * dy / d

        # Begrenzing totale afstotingssnelheid
        rep_spd = math.sqrt(rep_x * rep_x + rep_y * rep_y)
        if rep_spd > MAX_REPULSE_SPD:
            scale = MAX_REPULSE_SPD / rep_spd
            rep_x *= scale
            rep_y *= scale

        # Transformeer afstotingsvector van wereldframe → robotframe
        # (robotframe: x vooruit, y links)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rep_vx =  rep_x * cos_t + rep_y * sin_t
        rep_vy = -rep_x * sin_t + rep_y * cos_t

        return vx_robot + rep_vx, vy_robot + rep_vy

    # ─── Hulpmethodes ─────────────────────────────────────────────────────────
    def get_obstacles(self) -> List[Tuple[float, float]]:
        """Geeft een kopie van de opgeslagen obstakelpunten (wereldframe [m])."""
        return list(self._obstacles)

    def clear(self) -> None:
        """Verwijdert alle opgeslagen obstakels."""
        self._obstacles.clear()

    def __len__(self) -> int:
        return len(self._obstacles)

    def __repr__(self) -> str:
        return f"ObstacleMap({len(self._obstacles)} obstakels)"
