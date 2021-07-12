"use strict";

MCMC.registerAlgorithm("CoordinateHitAndRun", {
    description: "Coordinate Hit&Run (With uniform line distribution)",

    about: () => {
        window.open("https://doi.org/10.1093/bioinformatics/btx052");
    },

    init: (self) => {
        self.coordinate = 0;
        let A;
        let b;
    },

    reset: (self) => {
        const startPoint = MultivariateNormal.getSample(self.dim);
        startPoint[0] = -.1;
        startPoint[1] = -.1;
        self.chain = [startPoint];
    },

    attachUI: (self, folder) => {
        folder.open();
    },


    step: (self, visualizer) => {
        self.coordinate = (self.coordinate + 1) % 2;
        console.log(`updating ${self.coordinate}`)
        if (!self.A) {
            self.A = visualizer.simulation.mcmc.constraints.getA();
            self.b = visualizer.simulation.mcmc.constraints.getB();
        }


        const slacks = self.b.subtract(self.A.multiply(self.chain.last()));
        const inverseDistances = self.A.col(self.coordinate).cwiseQuotient(slacks);
        const forwardDistance = 1. / inverseDistances.maxCoeff();
        const backwardDistance = 1. / inverseDistances.minCoeff();
        console.log('inverseDistances')
        console.log(inverseDistances)
        const step = zeros(self.dim, 1);
        step[self.coordinate] = Math.random() * (forwardDistance - backwardDistance) + backwardDistance;
        console.log('state')
        console.log(self.chain.last()[self.coordinate])
        console.log('step')
        console.log(step[self.coordinate])
        const proposal = self.chain.last().add(step);

        const logAcceptRatio = +self.logDensity(proposal) - self.logDensity(self.chain.last());

        visualizer.queue.push({
            type: "proposal",
            proposal: proposal,
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

