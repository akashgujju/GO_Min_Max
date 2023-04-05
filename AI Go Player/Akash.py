import os
import copy
import random
import numpy as np
from copy import deepcopy


class GO:
    def __init__(self, n):
        """
        Class to setup the board, make the moves and describe the rules of the mini (5x5) Go game.
        'X' -> Player - 1
        'O' -> PLayer - 2
        :param n: the board size
        """
        self.size = n
        self.X_move = True  # True means 'X' plays first, otherwise 'O' plays first
        self.died_pieces = set()  # a set to keep track of the died pieces once an opponent has made it's move
        self.n_move = 0  # variable to keep track of the number of current moves in the game
        self.max_move = (n * n) - 1  # maximum number of moves allowed as per the given rule book
        self.komi = n / 2  # initialize the Komi rule

        # initialize the previous and current board state
        self.previous_board = np.zeros((n, n)).astype(int)
        self.board = np.zeros((n, n)).astype(int)
   
    def setup_board(self, piece_type, previous_board, board):
        """
        Set up the current board status.
        :param previous_board: previous board state
        :param board: current board state
        :param piece_type: indicating player type. (1->'X', 2->'O')
        :return: None.
        """

        for i in range(self.size):
            for j in range(self.size):
                # check the previous and current state of the board and append the number of died pieces
                if (previous_board[i, j] == piece_type) and (board[i, j] != piece_type):
                    self.died_pieces.add((i, j))

        self.previous_board = previous_board
        self.board = board

    def compare_board(self, board1, board2):
        """
        function to compare states of boards
        :param board1: previous board state
        :param board2: new board state
        :return: True if previous and current state of the board matches
        """
        for i in range(self.size):
            for j in range(self.size):
                if board1[i, j] != board2[i, j]:
                    return False
        return True

    def copy_board_instance(self):
        """
        Crate a deep copy instance of the passed GO object

        :param: None.
        :return: the copied board instance.
        """
        return deepcopy(self)

    def update_current_board(self, new_board):
        """
        Update the current board state with new_board

        :param new_board: new board.
        :return: None.
        """
        self.board = new_board

    def find_neighbors(self, i, j):
        """
        Return all the neighbors of the passed stone index.

        :param i: row index
        :param j: column index
        :return: list of the neighbors
        """
        board = self.board
        neighbors = []

        # Add row neighbors
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))

        # Add column neighbors
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def find_neighbor_ally(self, i, j):
        """
        function to find neighbors with same piece type (pool of allies)

        :param i: row index of the piece
        :param j: column index of the piece
        :return: a list containing the neighboring allies
        """
        board = self.board
        get_neighbors = self.find_neighbors(i, j)  # get neighbors
        piece_allies = []

        # iterate through the neighbors and append the list if piece type matches
        for position in get_neighbors:
            if board[position[0], position[1]] == board[i, j]:
                piece_allies.append(position)
        return piece_allies

    def all_allies(self, i, j):
        """
        find_neighbor_ally function returns only the next neighbor allies , this
        one will search and returns the whole group of allies spread in the board for a passed index (i,j) values.

        :param i: row index of the piece
        :param j: column index of the piece
        :return: a list containing all the allies for the passed board index value
        """
        stack = [(i, j)]  # instantiate the stack wih the passed index values for DFS search
        ally_members = []  # list of all allied members
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            get_neighbor_allies = self.find_neighbor_ally(piece[0], piece[1])
            for ally in get_neighbor_allies:
                if (ally not in stack) and (ally not in ally_members):
                    stack.append(ally)
        return ally_members

    def find_liberty(self, i, j):
        """
        function to check the liberty rule of the Go game. If the stone/stone allies don't have a liberty left on the
        board they all die. Basically if there is empty space around a piece/it's allies, it has liberty. It is also
        important to check liberty rule while placing the stone on the board in order to avoid suicide move.

        :param i: row index
        :param j: column index
        :return: a boolean value(True/False) indicating the liberty status.
        """
        board = self.board
        get_ally_members = self.all_allies(i, j)
        for member in get_ally_members:
            get_neighbors = self.find_neighbors(member[0], member[1])
            for piece in get_neighbors:
                # if any of the neighboring allies have liberty return TRUE
                if board[piece[0], piece[1]] == 0:
                    return True
        # Return FALSE if neither the piece or it's adjoining allies have liberty left on the board
        return False

    def find_dead_pieces(self, piece_type):
        """
        function to check if the positions corresponding to the passed stone type has liberty left on the board.
        If no liberty left, add them to the list of died pieces.

        :param piece_type
        :return: a list containing the indices of dead pieces.
        """
        board = self.board
        get_dead_pieces = []

        # iterate over the current board status, find the intended piece and check if it has liberty left on the board.
        # If no liberty left append the index to dead pieces

        for i in range(len(board)):
            for j in range(len(board)):
                if board[i, j] == piece_type:
                    if not self.find_liberty(i, j):
                        get_dead_pieces.append((i, j))
        return get_dead_pieces

    def remove_dead_pieces(self, piece_type):
        """
        function to remove the dead pieces from the board once found using the find_dead_pieces function.
        Removing dead pieces = place (0,0) at those particular indices

        :param piece_type: 1 or 2
        :return: index of dead pieces.
        """

        get_dead_pieces = self.find_dead_pieces(piece_type)
        if not get_dead_pieces:
            return []
        else:
            board = self.board
            for piece in get_dead_pieces:
                board[piece[0], piece[1]] = 0
            self.update_current_board(board)
            return get_dead_pieces

    def valid_move(self, i, j, piece_type):
        """
        function to check whether a move is valid for the given piece type.
        It checked both the Liberty and KO Rule.

        :param i: row index
        :param j: column index
        :param piece_type: 1 or 2.
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board

        # first check if the coordinates are bounded within the board range
        if not ((i >= 0) and (i < len(board))):
            return False

        if not (j >= 0) and (j < len(board)):
            return False

        # Secondly check if the index is already occupied
        if board[i, j] != 0:
            return False

        # Create a copy of the whole board for valid place testing
        temp_go = self.copy_board_instance()
        temp_board = temp_go.board

        # Check 1: Place the piece and check if the place has liberty
        temp_board[i, j] = piece_type
        temp_go.update_current_board(temp_board)
        if temp_go.find_liberty(i, j):
            return True

        # If no liberty found, then remove the dead pieces of the opponent if any and check liberty again
        temp_go.remove_dead_pieces(3 - piece_type)
        if not temp_go.find_liberty(i, j):
            return False

        # Check 2: Check KO Rule
        else:
            if self.died_pieces and (self.compare_board(self.previous_board, temp_go.board)):
                return False
        return True

    def play_move(self, i, j, piece_type):
        """
        Place the move on the Go Board

        :param i: row index
        :param j: column index
        :param piece_type: 1 or 2
        :return: boolean indicating whether the piece is successfully placed on the board or not
        """
        board = self.board

        # check if the move is valid
        valid_placement = self.valid_move(i, j, piece_type)
        if not valid_placement:
            return False

        '''if the move is valid, replace the previous board status with current board status and update the current 
           status with the new valid move '''
        self.previous_board = deepcopy(board)

        # place the piece on the board
        board[i, j] = piece_type

        self.update_current_board(board)

        return True

    def game_over(self, action="MOVE"):
        """
        Check if the game continues or end
        Two Conditions for the game to end:

        # Case 1: max moves reached
        # Case 2: both the players pass the move.

        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        """

        # n_move : number of current moves
        # max_move : maximum number of moves set for the board

        if (self.n_move >= self.max_move) or\
                ((self.compare_board(self.previous_board, self.board)) and (action == "PASS")):
            return True
        else:
            return False



class MinMax:
    def __init__(self, n):
        """
        Class to build the Min Max Agent with alpha-beta pruning
        :param n: size of the board
        """
        self.size = n
        self.type = 'min-max agent'
        self.piece_type = None

        self.max_depth = 2  # max depth of the min-max tree
        self.encode_state = None  # encode the current state of the board into a string variable

        self.possible_moves = []  # list of best possible moves available

    def read_input(self, path):
        """
        function to read the input file
        :param path: path to read the input file
        :return: piece_type, previous board status and current board status
        """
        with open(path, 'r') as f:
            lines = f.readlines()
            piece_type = int(lines[0])
            previous_board = [[int(char) for char in line.rstrip('\n')] for line in lines[1: self.size+1]]
            current_board = [[int(x) for x in line.rstrip('\n')] for line in lines[self.size+1: 2*self.size + 1]]

            return piece_type, previous_board, current_board

    def write_output(self, result, path="output.txt"):
        """
        function to record the agent's move
        :param result: selected best move
        :param path: output path
        :return: None
        """
        res = ""
        if result == "PASS":
            res = "PASS"
        else:
            res += str(result[0]) + ',' + str(result[1])

        with open(path, 'w') as f:
            f.write(res)

    def find_open_liberty(self, board_state, i, j):
        """
        function to find the open liberty for the passed index i,j. This value is used in the evaluation function.

        :param board_state: passed board state
        :param i: row index
        :param j:column index
        :return: open liberty on the board
        """
        board = board_state.board
        get_ally_members = board_state.all_allies(i, j)
        liberty_set = set()
        for member in get_ally_members:
            neighbors = board_state.find_neighbors(member[0], member[1])
            for guy in neighbors:
                if board[guy[0], guy[1]] == 0:
                    liberty_set.add(guy)

        open_liberty = len(liberty_set)
        return open_liberty

    def evaluation_function(self, board_state, player):
        """
        function to calculate the strength of the piece/player on the current board
        :param board_state: passed board state
        :param player: player type (1,2)
        :return: strength value
        """
        player_1 = 0  # pieces occupied by player_1
        player_2 = 0  # pieces occupied by player_2
        eval_player_1 = 0  # strength of the player_1 on the board
        eval_player_2 = 0  # strength of the player_2 on the board

        for i in range(self.size):
            for j in range(self.size):
                if board_state.board[i, j] == self.piece_type:
                    player_1 += 1
                    open_liberty_player_1 = self.find_open_liberty(board_state, i, j)
                    eval_player_1 += player_1 + open_liberty_player_1
                elif board_state.board[i, j] == 3 - self.piece_type:
                    player_2 += 1
                    open_liberty_player_2 = self.find_open_liberty(board_state, i, j)
                    eval_player_2 += player_2 + open_liberty_player_2

        # resulting evaluation value
        resultant_eval = eval_player_1 - eval_player_2
        if player == self.piece_type:
            return resultant_eval
        else:
            return -1 * resultant_eval

    def min_max_agent(self, agent_board, max_depth, alpha, beta):
        """
        function to find the best move possible for our agent
        :param agent_board: current board state
        :param max_depth: max depth the agent will search for the move
        :param alpha: alpha value
        :param beta: beta value
        :return: best moves
        """
        # list of best possible moves in the current scenario
        get_best_moves = []
        my_best = 0

        # out of all the available possible moves for the piece, play moves one by one and return best moves
        for move in self.possible_moves:
            next_board_state = copy.deepcopy(agent_board)
            next_board_state.play_move(move[0], move[1], self.piece_type)

            # Remove the dead pieces of opponent after our move
            next_board_state.died_pieces = next_board_state.remove_dead_pieces(3 - self.piece_type)

            # get the strength and evaluation value for the current move
            get_eval_value = self.evaluation_function(next_board_state, 3 - self.piece_type)

            get_score = self.min_max(next_board_state, max_depth,
                                     alpha, beta, get_eval_value, 3 - self.piece_type)

            curr_score = -1 * get_score
            # if current score is better than the previous make it as best move
            if (curr_score > my_best) or (not get_best_moves):
                my_best = curr_score
                alpha = my_best  # update alpha value
                get_best_moves = [move]
            # if current score is same as previous, append it to the list and later chose any random move
            elif curr_score == my_best:
                get_best_moves.append(move)

        return get_best_moves

    def min_max(self, curr_board, max_depth, alpha, beta, eval_value_temp, next_player):
        """
        function to implement the min_max algorithm with alpha-beta pruning
        :param curr_board: current board status
        :param max_depth: max-depth to search for the best move
        :param alpha: alpha value
        :param beta: beta value
        :param eval_value_temp: temp eval value
        :param next_player: opponent piece type
        :return: eval_value_temp
        """
        if max_depth == 0:
            return eval_value_temp

        best_temp = eval_value_temp

        # find possible moves for the opponent
        new_possible_moves = []
        for i in range(curr_board.size):
            for j in range(curr_board.size):
                if curr_board.valid_move(i, j, next_player):
                    new_possible_moves.append((i, j))

        # iterate through all valid moves
        for move in new_possible_moves:
            next_state = copy.deepcopy(curr_board)
            next_state.play_move(move[0], move[1], 3 - next_player)

            next_state.died_pieces = next_state.remove_dead_pieces(3 - self.piece_type)

            get_eval_value_temp = self.evaluation_function(next_state, 3 - next_player)
            get_score_temp = self.min_max(next_state, max_depth - 1,
                                          alpha, beta, get_eval_value_temp, 3 - next_player)

            # Maximizing Player (Our Agent)
            if next_player == self.piece_type:
                best_temp = max(get_score_temp, best_temp)
                alpha = max(best_temp, alpha)

                # Alpha Beta Pruning
                if beta <= alpha:
                    return best_temp

            # Minimizing player (Opponent Piece)
            elif next_player == 3 - self.piece_type:
                best_temp = min(get_score_temp, best_temp)
                beta = min(best_temp, beta)

                # Alpha Beta Pruning
                if beta <= alpha:
                    return best_temp

        return best_temp

    def get_input(self, go_board, piece):
        """
        function to get the agent's next move
        :param go_board: current go board
        :param piece: piece type (1->X or 2-> O)
        :return: PASS or best move
        """
        # set the player's/agent's piece type
        self.piece_type = piece

        # check if the game is still on
        if go_board.game_over():
            return 'Game Over!!'

        else:
            # first get all the possible moves available on the board
            self.possible_moves = []
            for i in range(go_board.size):
                for j in range(go_board.size):
                    if go_board.valid_move(i, j, self.piece_type):
                        self.possible_moves.append((i, j))

            # if no possible moves found return PASS
            if not self.possible_moves:
                return 'PASS'
            else:
                self.encode_state = ''.join(
                    [str(go_board.board[i, j]) for i in range(go_board.size) for j in range(go_board.size)])

                # if the board is empty and we are player-1 'X' just place the piece on (3,2)
                if self.encode_state == "0000000000000000000000000" and self.piece_type == 1:
                    action = (3, 2)
                    return action
                else:
                    # call the agent and find the next best move
                    max_depth = self.max_depth
                    action = self.min_max_agent(go_board, max_depth, alpha=-np.inf, beta=np.inf)

                    if not action:
                        return "PASS"
                    else:
                        action = random.choice(action)
                        return action


if __name__ == '__main__':
    board_size = 5
    set_path = os.path.join("input.txt")
    my_player = MinMax(board_size)
    game_board = GO(board_size)
    get_piece_type, get_previous_board, get_current_board = my_player.read_input(set_path)
    game_board.setup_board(get_piece_type, np.matrix(get_previous_board).astype(int),
                           np.matrix(get_current_board).astype(int))
    get_action = my_player.get_input(game_board, get_piece_type)
    my_player.write_output(get_action)
