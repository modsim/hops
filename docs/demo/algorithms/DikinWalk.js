"use strict";

function dikinEllipse(visualizer, x) {
    const A = visualizer.simulation.mcmc.constraints.getA();
    const b = visualizer.simulation.mcmc.constraints.getB();

    const diagonal = zeros(b.rows, 1);

    for (let i = 0; i < b.rows; ++i) {
        const val = 1. / ((b[i] - A.row(i).dot(x)) * (b[i] - A.row(i).dot(x)));
        diagonal[i] = val;
    }

    const D = A.transpose().multiply(diagonal.asDiagonal()).multiply(A);
    return D.llt_inverse();
}

MCMC.registerAlgorithm("DikinWalk", {
    description: "Dikin Walk",

    about: () => {
        window.open("https://www.jstor.org/stable/41412339 ");
    },

    init: (self) => {
        self.sigma = .5;
    },

    reset: (self) => {
        const startPoint = MultivariateNormal.getSample(self.dim);
        startPoint[0] = 0;
        startPoint[1] = 0;
        self.chain = [startPoint];
    },

    attachUI: (self, folder) => {
        folder.add(self, "sigma", 0.05, 2).step(0.05).name("Proposal &sigma;");
        folder.open();
    },


    step: (self, visualizer) => {
        const currentD = dikinEllipse(visualizer, self.chain.last());
        const proposalDist = new MultivariateNormal(self.chain.last(), currentD.scale(self.sigma * self.sigma));

        const proposal = proposalDist.getSample();
        const proposalD = dikinEllipse(visualizer, proposal);

        const logAcceptRatio =
            Math.log(Math.sqrt(proposalD.det()))
            - Math.log(Math.sqrt(currentD.det()))
            + self.logDensity(proposal)
            - self.logDensity(self.chain.last())

        ;

        visualizer.queue.push({
            type: "proposal",
            proposal: proposal,
            proposalCov: proposalDist.cov,
        });
        if (Math.random() < Math.exp(logAcceptRatio) && visualizer.simulation.isInteriorPoint(proposal)) {
            self.chain.push(proposal);
            visualizer.queue.push({type: "accept", proposal: proposal});
        } else {
            self.chain.push(self.chain.last());
            visualizer.queue.push({type: "reject", proposal: proposal});
        }
    },

});

