from typing import Dict, List, Tuple, Optional
from collections import Counter

from training_logger import LOGGER


class DataValidator:
    """
    Validador de dados com checagens rigorosas.
    
    Checagens incluídas:
    - Consistência de rodada (API vs partidas)
    - Duplicatas em /partidas
    - Abreviações desconhecidas no arquivo de odds
    - Jogos duplicados no OddsCasas.txt
    - Odds inválidas (<=1.01 ou não-numéricas)
    - Jogos na API sem odds / odds sem jogos na API
    - Clubes com contagem de partidas != 1
    - Atletas sem partida na rodada
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(
        self,
        status: Dict,
        partidas: List[Dict],
        clubes: Dict,
        odds_pairs: Dict[str, Tuple[float, float, float]],
        odds_pair_counts: Optional[Dict[str, int]] = None,
        odds_unknown_abbrs: Optional[List[str]] = None,
        atletas: Optional[List[Dict]] = None,
        rodada_partidas: Optional[int] = None,
        strict: bool = True,
    ) -> bool:
        self.errors = []
        self.warnings = []

        self._validate_rodada(status, partidas, rodada_partidas, strict=strict)
        self._validate_matches(partidas, clubes, odds_pairs, odds_pair_counts, odds_unknown_abbrs, strict=strict)

        if atletas is not None:
            self._validate_atletas(atletas, partidas, strict=strict)

        if self.errors:
            LOGGER.error("ERROS DE VALIDAÇÃO (críticos):")
            for e in self.errors:
                LOGGER.error(f"   • {e}")

        if self.warnings:
            LOGGER.warning("AVISOS:")
            for w in self.warnings:
                LOGGER.warning(f"   • {w}")

        return len(self.errors) == 0

    def _validate_rodada(self, status: Dict, partidas: List[Dict], rodada_partidas: Optional[int], strict: bool):
        rodada_atual = status.get("rodada_atual", None)

        if rodada_partidas is None and partidas:
            rodada_partidas = partidas[0].get("rodada", None)

        if rodada_atual is None:
            msg = "GET /mercado/status não retornou 'rodada_atual'."
            (self.errors if strict else self.warnings).append(msg)
            return

        if rodada_partidas is None:
            msg = f"GET /partidas sem 'rodada' (rodada_partidas=None), mas /mercado/status.rodada_atual={rodada_atual}."
            (self.errors if strict else self.warnings).append(msg)
            return

        if int(rodada_partidas) != int(rodada_atual):
            msg = f"Inconsistência de rodada: /partidas.rodada={rodada_partidas} mas /mercado/status.rodada_atual={rodada_atual}."
            (self.errors if strict else self.warnings).append(msg)

    def _validate_matches(
        self,
        partidas: List[Dict],
        clubes: Dict,
        odds_pairs: Dict[str, Tuple[float, float, float]],
        odds_pair_counts: Optional[Dict[str, int]],
        odds_unknown_abbrs: Optional[List[str]],
        strict: bool,
    ):
        api_pairs_list: List[str] = []
        missing_club_abbr: List[str] = []

        for p in partidas:
            home_id = str(p.get("clube_casa_id"))
            away_id = str(p.get("clube_visitante_id"))

            home_abbr = (clubes.get(home_id, {}) or {}).get("abreviacao")
            away_abbr = (clubes.get(away_id, {}) or {}).get("abreviacao")

            if not home_abbr:
                missing_club_abbr.append(f"clube_casa_id={home_id}")
                home_abbr = f"ID{home_id}"
            if not away_abbr:
                missing_club_abbr.append(f"clube_visitante_id={away_id}")
                away_abbr = f"ID{away_id}"

            api_pairs_list.append(f"{str(home_abbr).upper()}_{str(away_abbr).upper()}")

        api_pairs = set(api_pairs_list)
        odds_set = set(k.upper() for k in odds_pairs.keys())

        # Checagem de duplicatas em /partidas
        if len(api_pairs_list) != len(api_pairs):
            dup = [k for k, v in Counter(api_pairs_list).items() if v > 1]
            msg = f"Duplicatas em /partidas (mesmo HOME_AWAY repetido): {dup}"
            (self.errors if strict else self.warnings).append(msg)

        # Abreviações desconhecidas no arquivo de odds
        if odds_unknown_abbrs:
            unknown = sorted(set(a.upper() for a in odds_unknown_abbrs))
            msg = f"Odds contém abreviações desconhecidas: {unknown}"
            (self.errors if strict else self.warnings).append(msg)

        # Jogos duplicados no OddsCasas.txt
        if odds_pair_counts:
            dups = sorted([k for k, c in odds_pair_counts.items() if c > 1])
            if dups:
                msg = f"OddsCasas.txt tem jogos duplicados: {dups}"
                (self.errors if strict else self.warnings).append(msg)

        # Odds inválidas (<=1.01 ou não-numéricas)
        bad_odds = []
        for k, (oh, od, oa) in odds_pairs.items():
            try:
                oh, od, oa = float(oh), float(od), float(oa)
                if oh <= 1.01 or od <= 1.01 or oa <= 1.01:
                    bad_odds.append((k, oh, od, oa))
            except Exception:
                bad_odds.append((k, oh, od, oa))
        if bad_odds:
            msg = f"Odds inválidas (<=1.01 ou não-numéricas) em: {bad_odds[:5]}{'...' if len(bad_odds) > 5 else ''}"
            (self.errors if strict else self.warnings).append(msg)

        # Jogos na API sem odds
        missing_in_odds = api_pairs - odds_set
        # Odds sem jogos na API
        missing_in_api = odds_set - api_pairs

        if missing_in_odds:
            msg = f"Jogos na API mas NÃO estão no OddsCasas.txt: {sorted(missing_in_odds)}"
            (self.errors if strict else self.warnings).append(msg)

        if missing_in_api:
            msg = f"Jogos no OddsCasas.txt mas NÃO estão na API: {sorted(missing_in_api)}"
            (self.errors if strict else self.warnings).append(msg)

        # Clubes sem abreviação (sempre warning)
        if missing_club_abbr:
            msg = f"Clubes sem abreviação: {sorted(set(missing_club_abbr))}"
            self.warnings.append(msg)

    def _validate_atletas(self, atletas: List[Dict], partidas: List[Dict], strict: bool):
        clube_counts: Dict[int, int] = {}
        for p in partidas:
            h = p.get("clube_casa_id")
            a = p.get("clube_visitante_id")
            if h is not None:
                clube_counts[int(h)] = clube_counts.get(int(h), 0) + 1
            if a is not None:
                clube_counts[int(a)] = clube_counts.get(int(a), 0) + 1

        # Clubes com contagem de partidas != 1
        multi = sorted([cid for cid, cnt in clube_counts.items() if cnt != 1])
        if multi:
            msg = f"Clubes com contagem de partidas != 1 em /partidas: {multi}"
            (self.errors if strict else self.warnings).append(msg)

        # Atletas prováveis/dúvida sem partida
        target = [a for a in atletas if a.get("status_id") in (7, 2)]
        sem_partida = []
        for a in target:
            cid = a.get("clube_id")
            if cid is None or int(cid) not in clube_counts:
                sem_partida.append(f"{a.get('apelido', '?')} (clube_id={cid})")

        if sem_partida:
            self.warnings.append(f"{len(sem_partida)} atletas sem partida na rodada. Ex.: {sem_partida[:10]}")
