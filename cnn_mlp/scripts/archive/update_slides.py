replacement_text = r"""% -----------------------------------------------------------------------------
% SLIDE 24: Qualitative Evidence: CNN+MLP Flux Reconstruction
% -----------------------------------------------------------------------------
\begin{frame}{11a. Qualitative Evidence: CNN+MLP 2D Flux Reconstruction}
  \centering
  \includegraphics[width=0.74\textwidth, height=0.52\textheight, keepaspectratio]{figures/psi_samples_cnn_mlp.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Ground Truth vs. Predicted Field:}
      The predicted 2D poloidal flux field $\psi(R, Z)$ (center) matches the ground truth physics code EFIT (left) with high visual and topological fidelity.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Absolute Reconstruction Error:}
      Error heatmap (right) shows near-zero absolute discrepancy in the hot plasma core and across the critical outer magnetic boundary (LCFS).
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 25: Qualitative Evidence: CNN+MLP Training Dynamics & Loss Convergence
% -----------------------------------------------------------------------------
\begin{frame}{11b. Qualitative Evidence: CNN+MLP Training Convergence Dynamics}
  \centering
  \includegraphics[width=0.70\textwidth, height=0.50\textheight, keepaspectratio]{figures/loss_curves_cnn_mlp.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Clean Generalization:}
      Tuned Adam weight decay ($10^{-3}$) prevents memorization of training shot IDs. The validation loss curve closely tracks training loss throughout training.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Stable Convergence:}
      Loss steadily drops over 20 epochs ($\sim 1,500$ steps) before plateauing naturally. Early stopping saves the exact model with minimum validation loss.
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 26: Qualitative Evidence: CNN+LSTM Flux Reconstruction
% -----------------------------------------------------------------------------
\begin{frame}{11c. Qualitative Evidence: CNN+LSTM 2D Flux Reconstruction}
  \centering
  \includegraphics[width=0.74\textwidth, height=0.52\textheight, keepaspectratio]{figures/psi_samples_cnn_lstm.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Temporal Recurrent Mapping:}
      Reconstructs general plasma shape and elongation well, tracking core position across sequential 5 ms time-slices.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Boundary Diffuseness:}
      Higher error along the Last Closed Flux Surface (LCFS) due to hidden-state recurrent lag across static equilibrium slices ($\text{NRMSE} = 0.1969$).
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 27: Qualitative Evidence: CNN+LSTM Training Convergence Dynamics
% -----------------------------------------------------------------------------
\begin{frame}{11d. Qualitative Evidence: CNN+LSTM Training Convergence Dynamics}
  \centering
  \includegraphics[width=0.70\textwidth, height=0.50\textheight, keepaspectratio]{figures/loss_curves_cnn_lstm.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Recurrent Optimization Profile:}
      Training loss exhibits characteristic recurrent gradient dynamics with slight oscillations as hidden states $h_t$ adapt across shot sequences.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Validation Plateau:}
      Validation error reaches its optimum around step 1,100; recurrent parameters require higher memory overhead without reaching MLP precision.
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 28: Qualitative Comparison: CNN+MLP vs. CNN+LSTM
% -----------------------------------------------------------------------------
\begin{frame}{11e. Visual Model Comparison: CNN+MLP vs. CNN+LSTM}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \centering
      \textbf{\color{NavyBlue} CNN+MLP ($\text{NRMSE} = 0.1419$)}
      \vspace{2pt}

      \includegraphics[width=0.95\textwidth, height=0.40\textheight, keepaspectratio]{figures/psi_samples_cnn_mlp.png}

      \vspace{2pt}
      \footnotesize
      \textbf{Sharp Boundary Surfaces:} Reconstructs steep edge flux gradients accurately; zero spatial distortion.
    \end{column}

    \begin{column}{0.48\textwidth}
      \centering
      \textbf{\color{NavyBlue} CNN+LSTM ($\text{NRMSE} = 0.1969$)}
      \vspace{2pt}

      \includegraphics[width=0.95\textwidth, height=0.40\textheight, keepaspectratio]{figures/psi_samples_cnn_lstm.png}

      \vspace{2pt}
      \footnotesize
      \textbf{Temporal State Lag:} Captures elongation, but recurrent state accumulation causes slight edge contour blurring.
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 29: Sensor Robustness Failure Scenarios
% -----------------------------------------------------------------------------
\begin{frame}{12a. Sensor Robustness: The 6 Failure Scenarios}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Physical Noise \& Drift Scenarios}
      \begin{enumerate}
        \item \textbf{Gaussian Noise:} Pre-amplifier thermal noise ($\sigma \in \{0.05, 0.10, 0.20\}$).
        \item \textbf{Sensor Drift:} Integrator calibration drift over long discharge pulses.
        \item \textbf{Stuck ADC Probes:} Diagnostic signals frozen at upper/lower voltage limits.
      \end{enumerate}
    \end{column}

    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Hardware Loss \& Dropout Scenarios}
      \begin{enumerate}
        \setcounter{enumi}{3}
        \item \textbf{Random Channel Dropout:} Disconnected cables or probe failures ($p \in \{0.10, 0.25, 0.50\}$).
        \item \textbf{Category Ablation:} Entire diagnostic subsystems lost (e.g., losing all 76 pickup coils).
        \item \textbf{Disruption-Proximate Failure:} Diagnostic loss occurring immediately prior to a plasma termination.
      \end{enumerate}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 30: Diagnostic Criticality Ranking
% -----------------------------------------------------------------------------
\begin{frame}{12b. Diagnostic Criticality: Which Sensors Matter Most?}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Safety-Critical Diagnostic Channels}
      \begin{itemize}
        \item \textbf{Summary Plasma Current ($I_p$):}
          Losing the total current signal causes an immediate \textbf{$+140\%$ error explosion}, as the model cannot set the overall flux scale.
        \item \textbf{Saddle Flux Loops:}
          Losing radial boundary flux loops causes severe boundary shape distortion and shifts the estimated magnetic axis.
      \end{itemize}
    \end{column}

    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Highly Resilient Diagnostic Channels}
      \begin{itemize}
        \item \textbf{Individual Pickup Coils ($B_R, B_Z$):}
          Because 76 pickup coils surround the vessel, losing 1 to 5 coils causes only minor degradation ($<5\%$ NRMSE increase).
        \item \textbf{Takeaway:} High spatial sensor density provides inherent fault tolerance for local magnetic field measurements.
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 31: Diagnostic Criticality Ranking (Bar Chart)
% -----------------------------------------------------------------------------
\begin{frame}{12c. Empirical Sensor Criticality: NRMSE Surge Across Channels}
  \centering
  \includegraphics[width=0.74\textwidth, height=0.50\textheight, keepaspectratio]{figures/robustness_criticality.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Plasma Current ($I_p$) Criticality:}
      Losing $I_p$ surges test NRMSE by \textbf{$+140.3\%$} ($0.142 \to 0.341$), proving global current normalization is essential for physical flux scaling.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} High Pickup Redundancy:}
      Losing pickup coil groups increases error by only $\mathbf{<5\%}$, demonstrating that 76 distributed sensors provide strong physical fault tolerance.
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 32: Online Fault Mitigation & Imputation
% -----------------------------------------------------------------------------
\begin{frame}{12d. Online Fault Mitigation \& Imputation Strategies}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Baseline Imputation Methods}
      \begin{itemize}
        \item \textbf{Zero-Fill (No Imputation Baseline):}
          Missing values replaced with zeros.
          Catastrophic failure: the 2D flux boundary warps inward, destroying plasma position estimation.
        \item \textbf{Mean-Fill (Historical Training Mean):}
          Missing probes filled with global training means.
          Maintains numerical stability, but flattens physical flux gradients and blurs edge boundaries.
      \end{itemize}
    \end{column}

    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Forward-Fill (Carry Last Known Value)}
      \begin{itemize}
        \item \textbf{Physics-Informed Real-Time Mitigation:}
          Carries forward the last valid physical reading before sensor loss.
        \item \textbf{Why it Succeeds:}
          Magnetic fields and plasma currents are physically continuous ($L \frac{dI}{dt} \approx 0$ over microsecond intervals).
        \item \textbf{Result:}
          Limits NRMSE degradation to $<\!18\%$, successfully preserving plasma boundary geometry.
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 33: Online Mitigation Performance (Imputation Chart)
% -----------------------------------------------------------------------------
\begin{frame}{12e. Online Mitigation Performance: Imputation Comparison}
  \centering
  \includegraphics[width=0.74\textwidth, height=0.50\textheight, keepaspectratio]{figures/robustness_mitigation.png}

  \vspace{4pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Zero-Fill Collapse ($+148\%$ Error):}
      Under severe $50\%$ channel dropout ($p=0.50$), zero-fill causes NRMSE to surge to $0.352$, collapsing the estimated boundary inwards.
    \end{column}
    \begin{column}{0.48\textwidth}
      \footnotesize
      \textbf{\color{NavyBlue} Forward-Fill Resilience ($<18\%$ Degradation):}
      Forward-fill holds NRMSE at $\mathbf{0.167}$ even at $50\%$ dropout, preserving magnetic axis position and plasma cross-section shape.
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 34: Limitations & Future Scope
% -----------------------------------------------------------------------------
\begin{frame}{13. Limitations \& Future Scope}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Current Project Limitations}
      \begin{itemize}
        \item \textbf{Compute Budget Subset:}
          Models trained on 40 GB subset ($\approx 6.8\%$ of MAST archive); scaling to all 565 GB requires multi-node cluster compute.
        \item \textbf{Single-Device Spherical Geometry:}
          Evaluated on MAST (spherical tokamak); aspect-ratio physics differ from conventional tokamaks (e.g., JET, ITER).
      \end{itemize}
    \end{column}

    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} Future Research Directions}
      \begin{itemize}
        \item \textbf{Liquid Neural Networks (LNN / CfC):}
          Explore continuous-time neural ODE architectures to handle irregular sampling rates and severe multi-sensor dropouts naturally through adaptive time constants.
        \item \textbf{Cross-Machine Transfer Learning:}
          Fine-tune baseline weights across JET and ASDEX-Upgrade datasets.
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 35: Project Conclusions
% -----------------------------------------------------------------------------
\begin{frame}{14. Conclusions}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} 1. Accurate Sub-Millisecond Surrogate}
      \begin{itemize}
        \item Demonstrated that deep neural networks reconstruct 2D plasma equilibrium fields ($\psi(R, Z)$ on a $65 \times 65$ grid) with $<1\text{ ms}$ latency and $0.1419$ NRMSE ($\approx 86\%$ physical accuracy).
      \end{itemize}
      \vspace{4pt}
      \textbf{\color{NavyBlue} 2. Spatial Feedforward Dominance}
      \begin{itemize}
        \item Proved that spatial feedforward (\textbf{CNN+MLP}) outperforms recurrent models (\textbf{CNN+LSTM}) on Markovian equilibrium reconstruction ($0.1419$ vs $0.1969$ NRMSE), avoiding recurrent state drift.
      \end{itemize}
    \end{column}

    \begin{column}{0.48\textwidth}
      \textbf{\color{NavyBlue} 3. First 2D Sensor Failure Benchmark}
      \begin{itemize}
        \item Executed the first systematic sensor robustness study on 2D tokamak equilibrium fields, establishing an empirical criticality hierarchy across magnetic probe channels.
      \end{itemize}
      \vspace{4pt}
      \textbf{\color{NavyBlue} 4. Effective Real-Time Mitigation}
      \begin{itemize}
        \item Demonstrated that \textbf{forward-fill imputation} provides an effective, real-time mitigation against probe hardware failures, preventing boundary collapse.
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% SLIDE 36: Thank You / Q&A
% -----------------------------------------------------------------------------
\begin{frame}[plain]
  \centering
  \vspace{1.8cm}
  {\Huge \textbf{\color{NavyBlue} Thank You}}\\[12pt]
  {\large \textbf{\color{TealAccent} Questions \& Discussion}}\\[24pt]
  \textbf{Team 15:} Kalidasan B \& Nishanth D\\
  \vspace{4pt}
  \textbf{Supervisor:} Dr. Simi S\\
  \vspace{4pt}
  \small School of Computing, Amrita Vishwa Vidyapeetham
\end{frame}

\end{document}
"""

with open('PPT/Team_15_Final_Review.tex', 'r', encoding='utf-8') as f:
    text = f.read()

prefix = text.split('% SLIDE 24: Qualitative Evidence: CNN+MLP Flux Reconstruction')[0]
# remove trailing dashes before split marker if any
prefix = prefix.rstrip(' -\r\n\t') + '\n\n'

new_full_text = prefix + replacement_text

with open('PPT/Team_15_Final_Review.tex', 'w', encoding='utf-8') as f:
    f.write(new_full_text)

print("Successfully replaced tail with 36 slides.")
