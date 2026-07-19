# TBL physics model and validity regime

TBL uses SI units throughout: seconds, metres, radians per second,
counts per second, and dimensionless probabilities. The package contains two
different simulation layers. They should not be confused.

- `FockSimulator` evaluates coherent passive linear optics for small photon
  numbers. It is the layer for interference and output occupation probabilities.
- `DigitalTwin` is a chronological Monte Carlo model for source emission,
  hardware timing, routing, loss, control electronics, and detector time tags.
  Event paths do not coherently recombine in this layer.
- `CoherentTimeBinLoop` expands time bins into optical modes and constructs a
  subunitary transfer matrix, retaining interference between injections and
  round trips that arrive in the same bin.

## Gaussian internal mode

The temporal field is

\[
\psi(t)=\frac{1}{(2\pi\sigma_t^2)^{1/4}}
\exp\left[-\frac{(1-iC)(t-t_0)^2}{4\sigma_t^2}\right]
\exp[-i\omega_0(t-t_0)].
\]

Here `temporal_width` is the intensity standard deviation \(\sigma_t\), and
`chirp` is the dimensionless quadratic chirp \(C\). The angular-frequency
intensity width is

\[
\sigma_\omega=\frac{\sqrt{1+C^2}}{2\sigma_t}.
\]

The overlap of two unequal, delayed, detuned, chirped Gaussian modes is
evaluated by the analytic complex Gaussian integral. Jones-vector overlap is
included. `purity` is a phenomenological internal-state purity; the maximum
pair indistinguishability is modeled as \(\sqrt{p_1p_2}\). This captures a
measured HOM purity but is not a complete density matrix for arbitrary mixed
spectral states. For such states, provide a measured positive-semidefinite
`overlap_matrix` directly to `FockSimulator`.

Second-order dispersion multiplies the spectral field by
\(\exp(i\,\mathrm{GDD}\,\Omega^2/2)\). `Wavepacket.dispersed` analytically
updates width and chirp and conserves spectral width and normalization.

## Causal spontaneous-emission internal mode

Quantum dots, atoms, and other radiatively decaying emitters can instead use
the normalized one-sided exponential field

\[
\psi(t)=\Theta(t-t_0)\tau^{-1/2}
\exp\left[-\frac{t-t_0}{2\tau}\right]
\exp[-i\omega_0(t-t_0)].
\]

For `profile="exponential"`, `temporal_width` is the radiative **intensity
lifetime** \(\tau\), not a Gaussian standard deviation. The exact squared
overlap of equal-lifetime, equally detuned packets delayed by \(\Delta t\) is

\[
M(\Delta t)=\exp(-|\Delta t|/\tau),
\]

and at zero delay with carrier detuning \(\Delta\omega\) it is Lorentzian,

\[
M(\Delta\omega)=\frac{1}{1+(\Delta\omega\tau)^2}.
\]

Unequal exponential lifetimes and mixed Gaussian/exponential pairs use exact
one-sided analytic integrals; the latter is evaluated in a scaled complementary
error-function form to avoid overflow. Because an ideal causal exponential has
Lorentzian spectral intensity, its spectral variance diverges and
`spectral_width_angular` reports infinity. A nonzero quadratic spectral phase
does not remain in the exponential profile family, so `dispersed` rejects
nonzero GDD instead of returning an incorrectly re-parameterized pulse. Use a
measured spectral transfer model when dispersion of such an emitter matters.

## Exact partial distinguishability

For labeled input photons with spatial inputs \(r_j\), detected output list
\(o_j\), transfer amplitudes \(U_j\), and internal Gram matrix
\(S_{jk}=\langle\phi_j|\phi_k\rangle\), the probability is evaluated as

\[
P(\mathbf{s})=\frac{1}{\prod_m s_m!\,\mathcal N}
\sum_{\sigma\in S_n}
\left(\prod_j S_{j,\sigma(j)}\right)
\operatorname{perm}(M_\sigma),
\]

with

\[
(M_\sigma)_{jk}=U_k(o_j,r_k)U_{\sigma(k)}^*(o_j,r_{\sigma(k)}),
\qquad
\mathcal N=\operatorname{perm}
\left[S_{jk}\,\delta_{r_jr_k}\right].
\]

This retains the full pairwise overlap phases and reduces to exact bosonic and
fully distinguishable limits. It is factorial in photon number, so exact
partial distinguishability defaults to at most seven photons. A mean-overlap
approximation remains available only when explicitly requested.

When a circuit contains `SpectralMatrixComponent`, each labeled photon uses
the transfer matrix interpolated at its own wavelength. Measured values must be
passive. Extrapolation outside the calibrated band is rejected by default.

## Correlated pulsed source

`CorrelatedPhotonSource` uses a truncated zero/one/two-photon distribution:

\[
p_2=\frac{g^{(2)}(0)\mu^2}{2},\qquad
p_1=\mu-2p_2,\qquad p_0=1-p_1-p_2.
\]

The requested \(\mu\) and \(g^{(2)}(0)\) are therefore reproduced by the first
two factorial moments as long as these equations define non-negative
probabilities. Collection loss is binomial. On/off blinking is a two-state
Markov chain per excitation pulse. Spectral diffusion is a stationary
Ornstein-Uhlenbeck process with pulse-to-pulse correlation

\[
\rho=\exp[-T_{\rm rep}/\tau_c].
\]

This source truncation is not appropriate when three-photon emission is
non-negligible or when a thermal/SPDC photon-number distribution is required.

## Multimode SPDC/SFWM pair source

`SPDCSource` accepts either `K` equally occupied effective Schmidt modes or a
normalized measured weight spectrum \(\{\lambda_k\}\), with
\(\sum_k\lambda_k=1\), and mean pair number \(\mu\) per pump pulse. Each
Schmidt mode is an independent thermal mode with mean \(\mu\lambda_k\), giving
the probability-generating function

\[
G(z)=\prod_k\left[1+\mu\lambda_k(1-z)\right]^{-1}.
\]

The exact total pair distribution for measured weights is obtained by
convolving these geometric distributions. Its moments and reduced spectral
purity are

\[
\operatorname{Var}(n)=\mu+\mu^2\sum_k\lambda_k^2,\qquad
g^{(2)}=1+\sum_k\lambda_k^2,\qquad
\mathcal P=\sum_k\lambda_k^2,
\]

so \(K_{\rm eff}=1/\mathcal P\). For equal weights \(\lambda_k=1/K\), the
total pair count reduces to the negative-binomial distribution

\[
P(n)=\frac{\Gamma(n+K)}{\Gamma(K)n!}
\left(\frac{\mu}{K}\right)^n
\left(1+\frac{\mu}{K}\right)^{-(n+K)}.
\]

This expression is valid for real effective \(K\geq1\), reduces to a geometric
thermal distribution at \(K=1\), and approaches Poisson statistics as
\(K\rightarrow\infty\). Its independently verified moments are

\[
\operatorname{Var}(n)=\mu+\frac{\mu^2}{K},\qquad
g^{(2)}=1+\frac1K.
\]

For equally weighted Schmidt coefficients the reduced heralded photon purity is
\(\mathcal P=1/K\). TBL multiplies either measured or effective spectral purity
by any additional phenomenological packet purity. For threshold herald
efficiency \(\eta_h\), the equal-weight click probability is

\[
P(\mathrm{click})=1-
\left(1+\frac{\mu\eta_h}{K}\right)^{-K}.
\]

With a per-gate herald dark-click probability \(p_d\), the click probability
is instead

\[
P(h)=1-(1-p_d)
\left(1+\frac{\mu\eta_h}{K}\right)^{-K},
\qquad
P(n\mid h)\propto P(n)\left[1-(1-p_d)(1-\eta_h)^n\right].
\]

For nonuniform weights, replace the power-law no-click term by
\(\prod_k(1+\mu\lambda_k\eta_h)^{-1}\).

The conditional state therefore includes an explicit vacuum fraction from
false heralds. In `heralded_spdc_hom_scan`, \(\eta_h\) is idler collection
times herald-detector efficiency. Signal collection loss is an exact binomial
thinning before the interference beam splitter. For surviving inputs \(n,m\),
the second internal mode is decomposed into components parallel and orthogonal
to the first with binomial weights set by their squared overlap \(M\). Each
parallel sector uses exact two-mode Fock beam-splitter amplitudes, while the
orthogonal sector is convolved incoherently. Output click probabilities are

\[
P_j(\mathrm{click}\mid k)=1-(1-p_{d,j})(1-\eta_j)^k,
\]

so multipair contamination, collection loss, detector inefficiency, threshold
saturation, and output dark clicks all contribute to the predicted HOM floor.
The uniform overlap closure is exact for two pure internal modes. Applying it
to mixed multimode SPDC states is phenomenological because an effective
Schmidt number alone does not specify the full joint spectral amplitude.

The conditional pair distribution is computed explicitly, including its
single-pair and multipair fractions, with a checked infinite-tail tolerance.
Central wavelengths can optionally be constrained by
\(1/\lambda_p=1/\lambda_s+1/\lambda_i\). Arm collection losses are independent
Bernoulli channels after pair creation; pump timing jitter is common to both
arms and relative timing jitter is applied to the idler.

## Fiber loop and control

The round-trip survival probability is

\[
\eta_{\rm rt}=\eta_{\rm lumped}
10^{-[\alpha_{\rm dB/km}L_{\rm km}+L_{\rm insertion,dB}]/10}.
\]

The loop applies \(\mathrm{GDD}=\beta_2L\) on every pass. PMD is represented as
a zero-mean arrival-time perturbation with RMS
\(D_{\rm PMD}\sqrt{L}\). By default, photons traversing within the same
simulation time bin share the same PMD and phase realization, as required for
a common classical fiber environment; `shared_environment=False` is available
for explicitly independent ensemble sampling. Round-trip phase noise is accumulated, while
`PhaseDrift` implements a continuous Wiener process whose increment variance is
proportional to elapsed time. Events at the same time share the same phase.

EOM routing includes finite extinction ratio, insertion transmission, drive
noise, control latency, feed-forward latency/jitter, and a linear voltage rise.
The voltage-to-routing map is phenomenological; measured transfer curves can
instead be supplied with a callable control.

For coherent time-bin evolution, the external and circulating fields obey

\[
y_n=\sqrt{\eta_c}[\sqrt{1-R_n}\,x_n+i\sqrt{R_n}\,\ell_n],
\]

\[
\ell_{n+d}=e^{i\phi_n}\sqrt{\eta_c\eta_{\rm rt}}
[i\sqrt{R_n}\,x_n+\sqrt{1-R_n}\,\ell_n],
\]

where \(d\) is the round-trip delay in time bins. Evaluating this recurrence on
every basis input forms the external transfer matrix. Energy remaining in the
loop beyond the final simulated bin is reported separately from physical loss.

## SNSPD and time tagger

Photon, dark, and afterpulse candidates are processed in true chronological
order. For elapsed time \(\Delta t\) after the hard dead time, efficiency
recovery is

\[
R(\Delta t)=1-\exp[-(\Delta t-t_{\rm dead})/\tau_{\rm rec}].
\]

The photon avalanche probability is the wavelength-dependent system efficiency
times \(R\). Independent pixels have independent recovery clocks. Accepted
avalanches then receive fixed latency, Gaussian readout jitter, an optional
positive exponential jitter tail, and time-tagger quantization. Dark counts
are stationary Poisson candidates. Afterpulsing is a phenomenological branching
process and should be fitted to a specific detector if it matters.

## Calibration and uncertainty

HOM counts are fitted with Poisson maximum likelihood by default. Accidental
background must be measured or estimated independently because background and
intrinsic visibility are not jointly identifiable from a single HOM scan.
Fisher covariance and parametric Poisson bootstrap intervals are available.
Loss estimates include exact Clopper-Pearson intervals before detector-efficiency
correction. Loss position requires ordered physical taps or round-trip-resolved
counts; one input/output pair cannot identify location.

## Important limits

- Passive, linear optics only; no Kerr/Raman/Brillouin nonlinear propagation.
- No master-equation bath or arbitrary continuous-frequency joint spectral
  amplitude. SPDC accepts measured Schmidt weights, but weights alone do not
  encode Schmidt-mode shapes, phases, or signal/idler frequency correlations;
  supply measured overlap matrices when those details matter.
- Event-domain beam splitters sample paths. Use `CoherentTimeBinLoop` or a
  measured expanded-mode transfer matrix for recombining paths.
- Gaussian PMD, Wiener phase noise, OU spectral diffusion, and detector
  afterpulsing are calibrated stochastic models, not microscopic device models.
- Exact Fock calculations scale exponentially; exact general partial
  distinguishability additionally scales factorially.

All stochastic APIs accept a seed. Validation rejects non-passive matrices,
non-positive-semidefinite overlap matrices, invalid probability distributions,
and uncalibrated spectral extrapolation rather than silently normalizing them.

## References

1. H. J. Ryser, *Combinatorial Mathematics*, Mathematical Association of
   America (1963), chapter on permanents.
2. D. G. Glynn, “The permanent of a square matrix,” *European Journal of
   Combinatorics* **31**, 1887–1891 (2010),
   <https://doi.org/10.1016/j.ejc.2010.01.010>.
3. M. C. Tichy, “Sampling of partially distinguishable bosons and the relation
   to the multidimensional permanent,” *Physical Review A* **91**, 022316
   (2015), <https://doi.org/10.1103/PhysRevA.91.022316>.
4. C. K. Hong, Z. Y. Ou, and L. Mandel, “Measurement of subpicosecond time
   intervals between two photons by interference,” *Physical Review Letters*
   **59**, 2044–2046 (1987), <https://doi.org/10.1103/PhysRevLett.59.2044>.
5. C. Clopper and E. Pearson, “The use of confidence or fiducial limits
   illustrated in the case of the binomial,” *Biometrika* **26**, 404–413
   (1934), <https://doi.org/10.1093/biomet/26.4.404>.
6. A. Christ *et al.*, “Probing multimode squeezing with correlation
   functions,” *New Journal of Physics* **13**, 033027 (2011),
   <https://doi.org/10.1088/1367-2630/13/3/033027>.
