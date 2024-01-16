use chess::{Board, ChessMove, Color, Game, GameResult, MoveGen};

pub struct GameManager {
    game: Game,
}

impl GameManager {
    pub fn new() -> GameManager {
        GameManager { game: Game::new() }
    }

    pub fn do_move(&mut self, chess_move: ChessMove) -> (bool, Option<GameResult>) {
        let res = self.game.make_move(chess_move);

        (res, self.game.result())
    }

    pub fn get_moves(&self) -> MoveGen {
        MoveGen::new_legal(&self.game.current_position())
    }

    pub fn side_to_move(&self) -> Color {
        self.game.side_to_move()
    }

    pub fn result(&self) -> Option<GameResult> {
        self.game.result()
    }

    pub fn board(&self) -> Board {
        self.game.current_position()
    }

    pub fn declare_draw(&mut self) -> bool {
        self.game.declare_draw()
    }

    pub fn get_historic(&self) -> String {
        let actions = self.game.actions();
        let mut res: String = String::new();
        let mut cont = 1;
        
        for ac in actions {
            let s = match ac {
                chess::Action::MakeMove(cm) => cm.to_string(),
                chess::Action::OfferDraw(c) => {
                    match c {
                                    Color::Black => "Black Offer Draw",
                                    Color::White => "White Offer Draw"
                                }
                }.to_owned(),
                chess::Action::AcceptDraw => "Accept Draw".to_owned(),
                chess::Action::DeclareDraw => "Declare Draw".to_owned(),
                chess::Action::Resign(c) => {
                    match c {
                        Color::White => "White resign",
                        Color::Black => "Black resign",
                    }
                }.to_owned(),
            };
            res.push_str(&format!("{}:{:?}, ", cont, s));
            cont += 1;
        }

        res
    }
}
