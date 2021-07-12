"use strict";

MCMC.registerConstraints("Polytope", {
  description: "Polytope Constraints",
  ub: 1,
  lb_x: -2,
  lb_y:  -1.5,

  init: (self) => {
  },

  getA() {
    return matrix([[1, 1], [-1, 0], [0, -1]], 3, 2);
  },

  getB() {
    return matrix([[this.ub], [-this.lb_x], [-this.lb_y]], 3, 1);
  },

  getVertices() {
    return [
      [[this.lb_x, this.lb_y], [this.ub-this.lb_y, this.lb_y]],
      [[this.lb_x, this.ub-this.lb_x], [this.ub-this.lb_y, this.lb_y]],
      [[this.lb_x, this.lb_y], [this.lb_x, this.ub-this.lb_x]],
    ];
  },
});
