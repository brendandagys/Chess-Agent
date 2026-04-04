use pyo3::prelude::*;

#[pymodule]
mod chess_engine {
    use chess_engine::api as engine;
    use pyo3::prelude::*;

    /// A scored move returned by `get_top_moves`.
    #[pyclass(get_all)]
    #[derive(Debug, Clone)]
    pub struct ScoredMove {
        /// UCI move string (e.g. `"e2e4"`, `"e7e8q"`).
        pub mv: String,
        /// Evaluation from the side-to-move's perspective, in pawns.
        /// Positive = good for the side to move.
        pub score: f64,
    }

    #[pymethods]
    impl ScoredMove {
        fn __repr__(&self) -> String {
            format!("ScoredMove(mv='{}', score={})", self.mv, self.score)
        }
    }

    /// Return the top `n` moves for the given FEN position, ranked by engine
    /// evaluation (best first). Each move is searched to a fixed shallow depth
    /// so scores are directly comparable across moves.
    #[pyfunction]
    pub fn get_top_moves(fen: &str, n: usize) -> PyResult<Vec<ScoredMove>> {
        engine::get_top_moves(fen, n)
            .map(|moves| {
                moves
                    .into_iter()
                    .map(|m| ScoredMove {
                        mv: m.mv,
                        score: m.score,
                    })
                    .collect()
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Apply a sequence of UCI moves (e.g. `["e2e4", "e7e5"]`) to the position
    /// given by `fen` and return the resulting FEN string.
    #[pyfunction]
    pub fn apply_moves(fen: &str, moves: Vec<String>) -> PyResult<String> {
        let moves_ref: Vec<&str> = moves.iter().map(String::as_str).collect();
        engine::apply_moves(fen, &moves_ref).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Return a static evaluation of the position in pawns from White's
    /// perspective. Positive = white is better. Uses quiescence search to
    /// resolve captures but does not do a full tree search.
    #[pyfunction]
    pub fn evaluate_position(fen: &str) -> PyResult<f64> {
        engine::evaluate_position(fen).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Return all legal moves in the position as UCI strings (e.g. `"e2e4"`).
    #[pyfunction]
    pub fn get_legal_moves(fen: &str) -> PyResult<Vec<String>> {
        engine::get_legal_moves(fen).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Return whether `square` (e.g. `"e4"`) is attacked by `by_color`
    /// (`"white"` / `"w"` or `"black"` / `"b"`).
    #[pyfunction]
    pub fn is_square_attacked(fen: &str, square: &str, by_color: &str) -> PyResult<bool> {
        engine::is_square_attacked(fen, square, by_color)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }
}
