"use strict";

MCMC.registerConstraints("NoConstraints", {
  description: "No Constraints",

  init: (self) => {
  },

  getA() {
    return matrix([0, 0], 1, 2);
  },

  getB() {
    return matrix([100, 100], 1, 1);
  },

  getVertices() {
    return [];
  },
});
