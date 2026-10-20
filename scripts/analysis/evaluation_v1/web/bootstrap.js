/* Standalone-page adapter. The reusable component has no fixed document IDs. */
(function () {
  "use strict";
  const host = document.getElementById("kai-evaluation");
  const source = document.getElementById("kai-evaluation-data");
  if (!host || !source) return;
  try {
    const data = JSON.parse(source.textContent);
    host.replaceChildren();
    window.kaiEvaluationView = window.KaiEvaluationDashboard.mount(host, data);
  } catch (error) {
    host.replaceChildren();
    const message = document.createElement("p");
    message.style.padding = "2rem";
    message.textContent = "图表未能加载。请查看静态审计报告。";
    const link = document.createElement("a");
    link.href = "audit.html";
    link.textContent = "打开审计报告";
    message.append(" ", link);
    host.append(message);
    console.error("Evaluation dashboard could not initialize", error);
  }
}());
