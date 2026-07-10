# Theory Folder Rules

## No obvious theorems

Every theorem must earn its place. If a result is an immediate consequence of the definitions (e.g., "two expectations over the same uniform distribution are equal"), state it as a one-line observation in prose — not as a numbered theorem with a proof block. A theorem is warranted only when the conclusion is non-obvious or requires a real argument.

Before writing a theorem, ask: "Would a reader who understands the definitions be surprised by this claim?" If no, it belongs in a remark or setup paragraph, not a theorem environment.

## Justify every assumption

Don't just state "assume β-smooth" or "assume Lipschitz." Explain why the assumption is reasonable for the actual model and task (e.g., "the loss is β-smooth because the softmax output is bounded and differentiable"). An assumption without justification makes the result feel disconnected from practice.

## Concrete takeaway after every theorem

After the proof, state in one or two sentences what the theorem means in practical terms for the problem at hand. A theorem that doesn't change how you think about the problem isn't pulling its weight.

## Tightness

If you prove a bound or a gap, show it's tight: provide a matching lower bound or a concrete example that hits the bound. Without tightness, the reader can't tell whether the gap is real or an artifact of loose analysis.

## Show assumptions are necessary

For each key assumption, either construct a counterexample showing the result fails without it, or explicitly note that removing it is open. This is what separates a meaningful theorem from a vacuous one.

## Proof intuition before formalism

Before each proof, include a one-sentence "why this is true" intuition. If you can't state the intuition in plain language, the theorem may be mechanical rather than insightful.
