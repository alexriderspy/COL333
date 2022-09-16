from wrapt_timeout_decorator import *


def run_solver_with_timeout(solver, input_string, time_out=-1):
    if time_out <= 0:
        solver.search(input_string)
        return

    @timeout(time_out)
    def fn():
        solver.search(input_string)
    fn()
    return
