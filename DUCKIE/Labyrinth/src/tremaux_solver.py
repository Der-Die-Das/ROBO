class Junction:
    def __init__(self, junc_id, cum_distance, available):
        self.id = junc_id
        self.cum_distance = cum_distance  # in metres
        self.marks = {direction: 0 for direction in available}

    def update_directions(self, available):
        for d in available:
            if d not in self.marks:
                self.marks[d] = 0

class TremauxSolver:
    """
    A Trémaux's algorithm implementation for maze solving.
    
    Call junction_reached(distance, available_directions)
    where:
      - distance: metres traveled since the last junction.
      - available_directions: list of cardinal directions (e.g., ['north', 'east', 'west']).
    The method returns the chosen direction.
    """
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.junctions = {}  # mapping id to Junction objects
        self.current_junction_id = 0
        self.next_id = 1
        self.last_direction = None  # last chosen direction
        
        # Define opposites to infer the entrance direction.
        self.opposites = {"north": "south", "south": "north", "east": "west", "west": "east"}
        
        # Create starting junction with cumulative distance 0.
        start = Junction(0, 0.0, [])
        self.junctions[0] = start

    def _find_or_create_junction(self, candidate_distance, available):
        # Check if any existing junction is within the threshold.
        for j in self.junctions.values():
            if abs(j.cum_distance - candidate_distance) < self.threshold:
                j.update_directions(available)
                return j
        
        # Create a new junction if none is close enough.
        new_junc = Junction(self.next_id, candidate_distance, available)
        self.junctions[self.next_id] = new_junc
        self.next_id += 1
        return new_junc

    def junction_reached(self, distance, available):
        """
        Called when a junction is reached.
        Returns the chosen direction.
        """
        came_from = None if self.last_direction is None else self.opposites[self.last_direction]
        parent = self.junctions[self.current_junction_id]
        candidate_distance = parent.cum_distance + distance

        junction = self._find_or_create_junction(candidate_distance, available)
        if came_from is not None and came_from in junction.marks:
            junction.marks[came_from] += 1

        marks = junction.marks
        unmarked = [d for d in available if marks[d] == 0 and d != came_from]
        if unmarked:
            choice = unmarked[0]
            marks[choice] += 1
        elif all(marks[d] > 0 for d in available):
            if came_from is not None and marks[came_from] < 2:
                marks[came_from] += 1
                choice = came_from
            else:
                choice = min(available, key=lambda d: marks[d])
                marks[choice] += 1
        else:
            choice = min(available, key=lambda d: marks[d])
            marks[choice] += 1

        self.current_junction_id = junction.id
        self.last_direction = choice
        return choice