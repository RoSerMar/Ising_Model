class Coupled_Oscillators:
    def __init__(self, N, m=1.0, k_springs=1.0, left_wall_k=0.0, right_wall_k=0.0):
        self.N = N
        self.M = self._build_mass_matrix(m)
        self.K = self._build_K_matrix(k_springs, left_wall_k, right_wall_k)

    def _biuld_mass_matrix(self, m):
        return M

    def _biuld_K_matrix(self, k_springs, left_wall_k, right_wall_k):
        return K
    
    def solve_coupled_system_linear_Euler(self, x0=None, v0=None, t_max=50, num_points=1000):
        # x0: posições iniciais
        # v0: velocidades iniciais
        # t_max: tempo final da simulação
        # num_points: passo a considerar na resolução das equações

        # deve retornar os tempos em que foi calculada a solução,
        # e as posições e velocidades para cada tempo (ambas contidas em u = (x, v), por exemplo)
        # deve retornar os resultados obtidospelo método de Euler e pelo método do numpy
        # alternativamente pode ser criado um método para a solução por Euler e outro para numpy
        return sol_euler_t, sol_euler_u
    
    def solve_coupled_system_linear_numpy(self, x0=None, v0=None, t_max=50, num_points=1000):
        # x0: posições iniciais
        # v0: velocidades iniciais
        # t_max: tempo final da simulação
        # num_points: passo a considerar na resolução das equações

        # deve retornar os tempos em que foi calculada a solução,
        # e as posições e velocidades para cada tempo (ambas contidas em u = (x, v), por exemplo)
        # deve retornar os resultados obtidospelo método de Euler e pelo método do numpy
        # alternativamente pode ser criado um método para a solução por Euler e outro para numpy
        return sol_t, sol_u
