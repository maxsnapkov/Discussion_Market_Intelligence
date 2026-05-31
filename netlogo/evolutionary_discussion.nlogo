extensions [csv]

globals [
  payoff-file
  initial-distribution-file
  output-file
  prob-revision
  noise
  alpha
  weighted-mode?
  n-steps
  random-seed-value
  payoff-matrix
  payoff-vector
  initial-distribution
  n-of-strategies
  initialized?
]

breed [players player]

players-own [
  strategy
  strategy-after-revision
  payoff
]

to setup
  ; не используется clear-all, так как NetLogo сбрасывает глобальные параметры
  ; параметры передаются из BehaviorSpace и Python конвейера
  clear-turtles
  clear-patches
  clear-drawing
  set initialized? false
  normalize-parameters
  if random-seed-value != 0 [ random-seed random-seed-value ]
  load-payoffs
  load-initial-distribution
  setup-players
  write-output-header
  reset-ticks
  set initialized? true
  export-distribution 0
end

to normalize-parameters
  if not is-number? n-steps [ set n-steps 100 ]
  if n-steps <= 0 [ set n-steps 100 ]

  if not is-number? prob-revision [ set prob-revision 0.03 ]
  if prob-revision < 0 [ set prob-revision 0 ]
  if prob-revision > 1 [ set prob-revision 1 ]

  if not is-number? noise [ set noise 0.0 ]
  if noise < 0 [ set noise 0 ]
  if noise > 1 [ set noise 1 ]

  if not is-number? alpha [ set alpha 0.5 ]
  if alpha < 0 [ set alpha 0 ]
  if alpha > 1 [ set alpha 1 ]

  if not is-number? random-seed-value [ set random-seed-value 0 ]

  if not is-string? output-file [ set output-file "netlogo_history.csv" ]
  if output-file = "" [ set output-file "netlogo_history.csv" ]
end

to-report truthy? [value]
  if value = true [ report true ]
  if value = false [ report false ]
  if value = 1 [ report true ]
  if value = 0 [ report false ]
  if is-string? value [
    if member? value ["true" "True" "TRUE" "1" "yes" "Yes" "YES"] [ report true ]
    if member? value ["false" "False" "FALSE" "0" "no" "No" "NO"] [ report false ]
  ]
  report false
end

to-report usable-file-path? [value]
  if not is-string? value [ report false ]
  if value = "" [ report false ]
  report file-exists? value
end

to load-payoffs
  ifelse usable-file-path? payoff-file [
    ifelse truthy? weighted-mode? [
      let rows csv:from-file payoff-file
      if empty? rows [ error (word "payoff-file is empty: " payoff-file) ]
      set payoff-vector map [ row -> first row ] rows
      set n-of-strategies length payoff-vector
    ] [
      set payoff-matrix csv:from-file payoff-file
      if empty? payoff-matrix [ error (word "payoff-file is empty: " payoff-file) ]
      set n-of-strategies length payoff-matrix
    ]
  ] [
    ; резерв для демонстрационного режима интерфейса
    ifelse truthy? weighted-mode? [
      set payoff-vector [1.0 1.1 0.9]
      set n-of-strategies length payoff-vector
    ] [
      set payoff-matrix [[1.0 1.1 0.9] [0.9 1.0 1.2] [1.1 0.8 1.0]]
      set n-of-strategies length payoff-matrix
    ]
  ]
end

to load-initial-distribution
  ifelse usable-file-path? initial-distribution-file [
    let rows csv:from-file initial-distribution-file
    if empty? rows [ error (word "initial-distribution-file is empty: " initial-distribution-file) ]
    set initial-distribution first rows
  ] [
    ; резерв при отсутствии CSV в демонстрационном режиме
    if n-of-strategies <= 0 [ set n-of-strategies 3 ]
    let total 1000
    let base floor (total / n-of-strategies)
    set initial-distribution n-values n-of-strategies [ base ]
    let remaining-count total - sum initial-distribution
    let idx 0
    while [idx < remaining-count] [
      set initial-distribution replace-item idx initial-distribution ((item idx initial-distribution) + 1)
      set idx idx + 1
    ]
  ]

  if length initial-distribution != n-of-strategies [
    error (word "Initial distribution length " length initial-distribution
      " does not match strategy count " n-of-strategies)
  ]
end

to setup-players
  let idx 0
  foreach initial-distribution [ count-value ->
    create-players round count-value [
      set strategy idx
      set strategy-after-revision idx
      set payoff 0
      setxy random-xcor random-ycor
      set color 15 + ((idx * 10) mod 125)
    ]
    set idx idx + 1
  ]
  if not any? players [ error "No players were created" ]
end

to go
  ; защита интерфейса от запуска go до setup
  ; BehaviorSpace в headless режиме сначала вызывает setup
  if not initialized? [ setup ]
  if ticks >= n-steps [ stop ]
  ask players [ update-payoff ]
  ask players [ maybe-revise-strategy ]
  ask players [ apply-revision ]
  tick
  export-distribution ticks
end

to update-payoff
  let mate one-of other players
  if mate = nobody [
    set payoff 1
    stop
  ]
  ifelse truthy? weighted-mode? [
    let my-payoff item strategy payoff-vector
    let mate-payoff item ([strategy] of mate) payoff-vector
    set payoff (alpha * my-payoff) + ((1 - alpha) * mate-payoff)
  ] [
    set payoff item ([strategy] of mate) (item strategy payoff-matrix)
  ]
end

to maybe-revise-strategy
  set strategy-after-revision strategy
  if random-float 1 < prob-revision [
    ifelse random-float 1 < noise [
      set strategy-after-revision random n-of-strategies
    ] [
      let observed one-of other players
      if observed != nobody [
        if [payoff] of observed > payoff [
          set strategy-after-revision [strategy] of observed
        ]
      ]
    ]
  ]
end

to apply-revision
  set strategy strategy-after-revision
end

to write-output-header
  if not is-string? output-file [ set output-file "netlogo_history.csv" ]
  if output-file = "" [ set output-file "netlogo_history.csv" ]
  if file-exists? output-file [ file-delete output-file ]
  file-open output-file
  let count-cols n-values n-of-strategies [ i -> word "strategy_" i ]
  let prop-cols n-values n-of-strategies [ i -> word "prop_" i ]
  file-print csv:to-row (sentence (list "step") (sentence count-cols prop-cols))
  file-close
end

to export-distribution [step-number]
  if not is-string? output-file [ set output-file "netlogo_history.csv" ]
  if output-file = "" [ set output-file "netlogo_history.csv" ]
  file-open output-file
  let total count players
  let counts n-values n-of-strategies [ i -> count players with [strategy = i] ]
  let props map [ c -> c / total ] counts
  file-print csv:to-row (sentence (list step-number) (sentence counts props))
  file-close
end
@#$#@#$#@
GRAPHICS-WINDOW
210
10
650
451
-1
-1
13.1
1
10
1
1
1
0
1
1
1
-16
16
-16
16
0
0
1
ticks
30.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
