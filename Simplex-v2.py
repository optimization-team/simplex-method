from Matrix import Matrix
from Function import Function

class Simplex2:
    '''
    Attributes
    ----------
    function: Function
        function to optimise

    matrix: Matrix
        matrix of constraints (assumed that all given in the form of inequalities)

    b: Matrix
        right hand side column vector (size n x 1)

    to_maximise: bool
        True - function has to be maximised
        False - function has to be minimised
    '''
    def __init__(self, function: Function, A: Matrix, b: Matrix, approximation: int | float,
                 to_maximise: bool = True):
        self.function = function
        self.A = A
        self.b = b
        self.z_i = None # after completing the loop in optimise(), it is the max value of the func

    #   !! requires calculating inverse of a matrix
    def is_optimal(self) -> bool:
        # TODO implement checker: (z_j-c_j)>=0  <=>  the solution is optimal
        return


    #TODO handle minimization case
    def optimise(self):
        #TODO initial step

        while not self.is_optimal():
            #TODO compute xB_i, B_i
            #TODO compute cB_i, z_i
            #TODO define the entering var
            #TODO define the leaving var

        #TODO retrieve the vector x: z(x)=z_max based on the previous computations
