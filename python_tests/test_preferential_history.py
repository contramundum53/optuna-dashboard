from __future__ import annotations

import json
from typing import Callable
from typing import TYPE_CHECKING

from optuna_dashboard._preferential_history import _SYSTEM_ATTR_PREFIX_HISTORY
from optuna_dashboard._preferential_history import NewHistory
from optuna_dashboard._preferential_history import remove_history
from optuna_dashboard._preferential_history import report_history
from optuna_dashboard._preferential_history import restore_history
from optuna_dashboard._serializer import serialize_preference_history
from optuna_dashboard.preferential import create_study
from optuna_dashboard.preferential._system_attrs import _SYSTEM_ATTR_PREFIX_PREFERENCE

from .storage_supplier import parametrize_storages
from .storage_supplier import StorageSupplier


if TYPE_CHECKING:
    from optuna_dashboard._preferential_history import History


@parametrize_storages
def test_report_and_get_choices(storage_supplier: Callable[[], StorageSupplier]) -> None:
    with storage_supplier() as storage:
        study = create_study(storage=storage, n_generate=5)
        for _ in range(5):
            trial = study.ask()
            trial.suggest_float("x", 0, 1)

        study_id = study._study._study_id

        report_history(
            study_id=study_id,
            storage=storage,
            input_data=NewHistory(mode="ChooseWorst", candidates=[0, 1, 2], clicked=1),
        )
        report_history(
            study_id=study_id,
            storage=storage,
            input_data=NewHistory(mode="ChooseWorst", candidates=[0, 2, 3, 4], clicked=0),
        )
        history = serialize_preference_history(storage.get_study_system_attrs(study_id))
        sys_attrs = storage.get_study_system_attrs(study_id)
        assert len(history) == 2
        assert history[0]["candidates"] == [0, 1, 2]
        assert history[0]["clicked"] == 1
        preferences = sys_attrs[_SYSTEM_ATTR_PREFIX_PREFERENCE + history[0]["preference_id"]]
        assert len(preferences) == 2
        for i, (best, worst) in enumerate([(0, 1), (2, 1)]):
            assert len(preferences[i]) == 2
            assert preferences[i][0] == best
            assert preferences[i][1] == worst
        assert history[1]["candidates"] == [0, 2, 3, 4]
        assert history[1]["clicked"] == 0
        preferences = sys_attrs[_SYSTEM_ATTR_PREFIX_PREFERENCE + history[1]["preference_id"]]
        assert len(preferences) == 3
        for i, (best, worst) in enumerate([(2, 0), (3, 0), (4, 0)]):
            assert len(preferences[i]) == 2
            assert preferences[i][0] == best
            assert preferences[i][1] == worst


@parametrize_storages
def test_undo_redo_history(storage_supplier: Callable[[], StorageSupplier]) -> None:
    with storage_supplier() as storage:
        study = create_study(storage=storage, n_generate=5)
        for _ in range(5):
            trial = study.ask()
            trial.suggest_float("x", 0, 1)

        study_id = study._study._study_id

        def get_preferences_history(history_id: str) -> tuple[list[tuple[int, int]], History]:
            system_attrs = storage.get_study_system_attrs(study_id)
            history: History = json.loads(
                system_attrs.get(_SYSTEM_ATTR_PREFIX_HISTORY + history_id, "")
            )
            preference: list[tuple[int, int]] = system_attrs.get(
                _SYSTEM_ATTR_PREFIX_PREFERENCE + history["preference_id"], []
            )
            return preference, history

        history_id = report_history(
            study_id=study_id,
            storage=storage,
            input_data=NewHistory(mode="ChooseWorst", candidates=[0, 1, 2], clicked=1),
        )
        remove_history(study_id, storage, history_id)
        preference, history = get_preferences_history(history_id)
        assert history["mode"] == "ChooseWorst"
        assert history["candidates"] == [0, 1, 2]
        assert history["clicked"] == 1
        assert len(preference) == 0

        remove_history(study_id, storage, history_id)
        preference, history = get_preferences_history(history_id)
        assert len(preference) == 0

        restore_history(study_id, storage, history_id)
        preference, history = get_preferences_history(history_id)
        assert history["mode"] == "ChooseWorst"
        assert history["candidates"] == [0, 1, 2]
        assert history["clicked"] == 1
        assert len(preference) == 2
        for i, (best, worst) in enumerate([(0, 1), (2, 1)]):
            assert len(preference[i]) == 2
            assert preference[i][0] == best
            assert preference[i][1] == worst

        restore_history(study_id, storage, history_id)
        preference, history = get_preferences_history(history_id)
        assert len(preference) == 2
