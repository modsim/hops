"use strict";

MCMC.registerConstraints("BoxConstraints", {
  description: "Box Constraints (lb < x_i < ub)",
  ub_x: 3.5,
  lb_x:  -3.5,
  ub_y: 1.5,
  lb_y:  -1.5,

  init: (self) => {
  },

  getA() {
    return matrix([1, 0, 0, 1, -1, 0, 0, -1], 4, 2);
  },

  getB() {
    return matrix([this.ub_x, this.ub_y, -this.lb_x, -this.lb_y], 4, 1);
  },

  getVertices() {
    return [
      [[this.ub_x, this.ub_y], [this.ub_x, this.lb_y]],
      [[this.ub_x, this.lb_y], [this.lb_x, this.lb_y]],
      [[this.lb_x, this.lb_y], [this.lb_x, this.ub_y]],
      [[this.lb_x, this.ub_y], [this.ub_x, this.ub_y]],
    ];
  },
});