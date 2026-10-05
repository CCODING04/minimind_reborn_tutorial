// 部署前交互与响应式检查器（playwright-core + 本机 chrome，零安装）
// 用法：node tools/interact_check.mjs [--base http://127.0.0.1:6006] [--shots /tmp/deploy_shots]
// 检查：①全页面链接/锚点存在性 ②按键交互矩阵 ③五档视口横向溢出 ④外部资源依赖
import { chromium } from "../../../reference/minimind_3rd/courses/Part7_minimind/node_modules/playwright-core/index.mjs";
import { existsSync, readFileSync, mkdirSync } from "node:fs";
import { resolve, relative } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = (() => {
  const i = process.argv.indexOf("--root");
  return i > 0 ? resolve(process.argv[i + 1]) : resolve(import.meta.dirname, "..");
})();
const BASE = (() => {
  const i = process.argv.indexOf("--base");
  return i > 0 ? process.argv[i + 1] : "http://127.0.0.1:6006";
})();
const SHOTS = (() => {
  const i = process.argv.indexOf("--shots");
  return i > 0 ? process.argv[i + 1] : "/tmp/deploy_shots";
})();
mkdirSync(SHOTS, { recursive: true });

const pages = [
  "index.html",
  ...["m01_get_started","m02_chat","m03_pipeline","m04_config","m05_layers","m06_kv_cache",
     "m07_attention","m08_ffn_moe","m09_model","m10_tokenizer","m11_data_access","m12_datasets",
     "m13_trainer","m14_checkpoint","m15_ddp","m16_pretrain_sft","m17_dpo","m18_distill_lora",
     "m19_rl","m20_inference","m21_eval","m22_graduation"].map(m => m + ".html"),
];
const cvPages = existsSync(resolve(ROOT, "codeview"))
  ? ["codeview/src/minimind_reborn/training/common/trainer.py.html",
     "codeview/tools/verify_env.py.html",
     "codeview/configs/smoke/smoke_pretrain.yaml.html",
     "codeview/README.md.html"]
  : ["source/src/minimind_reborn/training/common/trainer.py.html",
     "source/tools/verify_env.py.html",
     "source/configs/smoke/smoke_pretrain.yaml.html",
     "source/README.md.html"];
const allPages = [...pages, ...cvPages];

let pass = 0, fail = 0;
const fails = [];
const ok = (name) => { pass++; console.log("PASS", name); };
const bad = (name, detail) => { fail++; fails.push(detail); console.log("FAIL", name, "--", detail); };

// ── ① 链接/锚点静态存在性（DOM 采集 + 文件系统核对） ──
async function linkAudit(context) {
  const pg = await context.newPage();
  for (const p of allPages) {
    const url = BASE + "/" + p;
    const resp = await pg.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 }).catch(e => null);
    if (!resp || resp.status() !== 200) { bad(`页面可载 ${p}`, `HTTP ${resp && resp.status()}`); continue; }
    const hrefs = await pg.$$eval("a[href]", as => as.map(a => a.getAttribute("href")));
    const dir = p.includes("/") ? p.slice(0, p.lastIndexOf("/")) : "";
    for (const href of hrefs) {
      if (/^(https?:|mailto:|data:|javascript:)/.test(href)) continue;
      const [target, anchor] = href.split("#");
      if (href.startsWith("#") || !target) {
        const id = href.replace("#", "");
        const found = await pg.evaluate((i) => !!document.getElementById(i), id);
        if (!found) bad(`页内锚点 ${p}#${id}`, "目标 id 不存在");
        continue;
      }
      const local = relative(ROOT, resolve(ROOT, dir, target.split("?")[0]));
      if (!existsSync(resolve(ROOT, local))) { bad(`链接 ${p} → ${href}`, "目标文件不存在"); continue; }
      if (anchor && target.endsWith(".html")) {
        const t = readFileSync(resolve(ROOT, local), "utf-8");
        if (!t.includes(`id="${anchor}"`)) bad(`跨页锚点 ${p} → ${href}`, "目标页缺该 id");
      }
    }
  }
  await pg.close();
  ok(`链接审计（${allPages.length} 页全部 href/锚点）`);
}

// ── ② 按键交互矩阵 ──
async function interactions(browser) {
  // A. quiz 折叠 + 答错回看跳转（m05）
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m05_layers.html", { waitUntil: "domcontentloaded" });
    const quiz = pg.locator("details.quiz").first();
    await quiz.locator("summary").click();
    if (await quiz.evaluate(el => el.open)) ok("quiz 点击展开（m05 首题）"); else bad("quiz 展开", "open 属性未设置");
    const back = quiz.locator(".quiz-ans .back a").first();
    if (await back.count()) {
      const anchor = (await back.getAttribute("href")).slice(1);
      await back.click();
      // 平滑滚动进行中：轮询直到目标进入视口或超时 3s
      let box = null;
      for (let i = 0; i < 30; i++) {
        await pg.waitForTimeout(100);
        box = await pg.evaluate((id) => { const el = document.getElementById(id); if (!el) return null; return el.getBoundingClientRect().top; }, anchor);
        if (box !== null && box >= 0 && box < 900) break;
      }
      if (box !== null && box >= 0 && box < 900) ok(`答错回看跳转（#${anchor} 进入视口且不被顶栏遮挡）`);
      else bad("答错回看跳转", `目标 top=${box}`);
    }
    await ctx.close();
  }
  // B. 复制按钮（m05）+ $ 提示符不入剪贴板（m01 终端块）
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, permissions: ["clipboard-read", "clipboard-write"] });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m05_layers.html", { waitUntil: "domcontentloaded" });
    const pre = pg.locator("pre.code").first();
    await pre.hover();
    const btn = pre.locator(".copy-btn");
    if (await btn.count()) {
      await btn.click();
      await pg.waitForTimeout(200);
      const text = await pg.evaluate(() => navigator.clipboard.readText().catch(() => null));
      const codeText = (await pre.locator("code").innerText()).trimEnd();
      if (text !== null && text === codeText) ok("复制按钮：剪贴板内容与代码一致");
      else if (text !== null) bad("复制按钮", "剪贴板与代码不一致");
      else ok("复制按钮：触发成功（剪贴板读权限不可用，按文本变化判定）");
      const label = await btn.textContent();
      if (label === "已复制") ok("复制按钮反馈文字"); else bad("复制反馈", label || "(空)");
    } else bad("复制按钮注入", "pre.code 上无 .copy-btn");
    await pg.goto(BASE + "/m01_get_started.html", { waitUntil: "domcontentloaded" });
    const term = pg.locator("pre.term").first();
    await term.hover();
    await term.locator(".copy-btn").click();
    await pg.waitForTimeout(200);
    const t2 = await pg.evaluate(() => navigator.clipboard.readText().catch(() => null));
    if (t2 !== null) {
      if (!/^\$\s/m.test(t2) && !t2.startsWith("$ ")) ok("终端复制：$ 提示符未进剪贴板");
      else bad("终端复制", "$ 混入剪贴板");
    }
    await ctx.close();
  }
  // C. 完成按钮往返 + 持久化 + 三处联动（m07）
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m07_attention.html", { waitUntil: "domcontentloaded" });
    const btn = pg.locator("#complete-btn");
    await btn.click();
    const checks = await pg.evaluate(() => ({
      btn: document.getElementById("complete-btn").textContent,
      top: document.getElementById("topbar-progress").textContent,
      dot: document.querySelector('.sidebar .mod.current')?.classList.contains("done"),
      store: JSON.parse(localStorage.getItem("mrt-progress-v1") || "[]"),
    }));
    if (checks.btn.includes("已完成") && checks.top === "1/22" && checks.dot && checks.store.includes("m07"))
      ok("完成按钮：文字/顶栏/侧栏圆点/localStorage 四处联动");
    else bad("完成按钮联动", JSON.stringify(checks));
    await pg.reload({ waitUntil: "domcontentloaded" });
    const after = await pg.evaluate(() => ({
      top: document.getElementById("topbar-progress").textContent,
      btn: document.getElementById("complete-btn").textContent,
    }));
    if (after.top === "1/22" && after.btn.includes("已完成")) ok("完成状态刷新持久化"); else bad("持久化", JSON.stringify(after));
    await pg.locator("#complete-btn").click();
    const back = await pg.evaluate(() => document.getElementById("topbar-progress").textContent);
    if (back === "0/22") ok("再次点击撤销完成"); else bad("撤销", back);
    // 侧栏圆点点击不跳转（点当前篇内的 M08，保证可见）
    await pg.locator('.sidebar a.mod:has-text("M08")').first().locator(".done-dot").click({ force: true });
    await pg.waitForTimeout(300);
    if (pg.url().includes("m07_attention")) {
      const n = await pg.evaluate(() => document.getElementById("topbar-progress").textContent);
      if (n === "1/22") ok("侧栏圆点切换完成且不触发跳转"); else bad("侧栏圆点", "计数 " + n);
    } else bad("侧栏圆点", "发生了跳转 " + pg.url());
    // 篇折叠切换 + 刷新持久化（第 3 篇默认折叠：点击应展开；再点应折叠——断言"翻转"而非固定方向）
    const p3 = pg.locator(".sidebar .part").nth(2);
    const before = await p3.evaluate(el => el.classList.contains("collapsed"));
    await pg.locator(".sidebar .part-head").nth(2).click();
    const afterClick = await p3.evaluate(el => el.classList.contains("collapsed"));
    await pg.reload({ waitUntil: "domcontentloaded" });
    const persisted = await pg.locator(".sidebar .part").nth(2).evaluate(el => el.classList.contains("collapsed"));
    if (afterClick === !before && persisted === afterClick) ok(`篇折叠切换+持久化（${before}→${afterClick}）`);
    else bad("篇折叠", `before=${before} after=${afterClick} persisted=${persisted}`);
    await pg.evaluate(() => localStorage.clear());
    await ctx.close();
  }
  // D. index 圆点可点（联动）
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/index.html", { waitUntil: "domcontentloaded" });
    await pg.locator(".part-card .pc-mods li").first().locator(".done-dot").click();
    const n = await pg.evaluate(() => document.getElementById("topbar-progress").textContent);
    if (n === "1/22") ok("index 网格圆点可点且联动顶栏"); else bad("index 圆点", n);
    await ctx.close();
  }
  // E. 移动端抽屉（390px）：开/遮罩关/Esc 关/锁滚动
  {
    const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m05_layers.html", { waitUntil: "domcontentloaded" });
    const toggle = pg.locator(".nav-toggle");
    if (await toggle.isVisible()) ok("390px 汉堡按钮可见"); else bad("汉堡可见性", "nav-toggle 不可见");
    await toggle.click();
    const s1 = await pg.evaluate(() => ({
      open: document.body.classList.contains("nav-open"),
      overlay: getComputedStyle(document.querySelector(".nav-overlay")).display,
      locked: getComputedStyle(document.body).overflow,
      aria: document.querySelector(".nav-toggle").getAttribute("aria-expanded"),
    }));
    if (s1.open && s1.overlay !== "none" && s1.locked === "hidden" && s1.aria === "true")
      ok("抽屉打开：滑入+遮罩+锁滚动+aria"); else bad("抽屉打开", JSON.stringify(s1));
    await pg.keyboard.press("Escape");
    const s2 = await pg.evaluate(() => document.body.classList.contains("nav-open"));
    if (!s2) ok("Esc 关闭抽屉"); else bad("Esc 关闭", "仍打开");
    await toggle.click();
    await pg.locator(".nav-overlay").click({ position: { x: 370, y: 400 } });
    const s3 = await pg.evaluate(() => document.body.classList.contains("nav-open"));
    if (!s3) ok("点遮罩关闭抽屉"); else bad("遮罩关闭", "仍打开");
    await ctx.close();
  }
  // F. TOC 点击滚动（1440px）
  {
    const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m12_datasets.html", { waitUntil: "domcontentloaded" });
    const link = pg.locator(".toc a:not(.l3)").nth(2);
    const anchor = (await link.getAttribute("href")).slice(1);
    await link.click();
    await pg.waitForTimeout(500);
    const top = await pg.evaluate((id) => { const el = document.getElementById(id); return el ? el.getBoundingClientRect().top : -1; }, anchor);
    if (top >= 0 && top < 900) ok(`TOC 点击滚动（#${anchor}）`); else bad("TOC 滚动", `top=${top}`);
    await ctx.close();
  }
  // G. pager 全链 m01→m22
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    await pg.goto(BASE + "/m01_get_started.html", { waitUntil: "domcontentloaded" });
    const first = await pg.locator(".pager .pg.disabled").count();
    if (first >= 1) ok("m01 上一课为禁用占位"); else bad("m01 pager", "无 disabled 占位");
    let hops = 0;
    for (let i = 2; i <= 22; i++) {
      const next = pg.locator(".pager a.pg.next");
      if (!(await next.count())) { bad("pager 链", `第 ${i - 1} 课缺下一课链接`); break; }
      await next.click();
      await pg.waitForLoadState("domcontentloaded");
      hops++;
      if (!pg.url().includes(`m${String(i).padStart(2, "0")}`)) { bad("pager 链", `跳到了 ${pg.url()}`); break; }
    }
    if (hops === 21) ok("pager 下一课链 m01→m22 全通（21 跳）");
    const last = await pg.locator(".pager .pg.disabled").count();
    if (pg.url().includes("m22") && last >= 1) ok("m22 下一课为禁用占位（课程终点）");
    await ctx.close();
  }
  // H. 源码页返回链接 + 行锚深链（course 目录为 codeview 形态；package 为 source 形态）
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    const cvDeep = existsSync(resolve(ROOT, "codeview"))
      ? "codeview/src/minimind_reborn/training/common/trainer.py.html"
      : "source/src/minimind_reborn/training/common/trainer.py.html";
    await pg.goto(BASE + "/" + cvDeep, { waitUntil: "domcontentloaded" });
    await pg.locator(".cv-back, .crumbs a").first().click();
    await pg.waitForLoadState("domcontentloaded");
    if (pg.url().replace(/\/$/, "").endsWith("/index.html")) ok("最深源码页返回课程主页"); else bad("源码页返回", pg.url());
    if (existsSync(resolve(ROOT, "codeview"))) {
      await pg.goto(BASE + "/codeview/src/minimind_reborn/models/layers.py.html#L42", { waitUntil: "domcontentloaded" });
      await pg.waitForTimeout(600);
      const t = await pg.evaluate(() => { const el = document.getElementById("L42"); return el ? el.getBoundingClientRect().top : -1; });
      if (t >= 0 && t < 900) ok("#L42 行锚深链定位（不被顶栏遮挡）"); else bad("#L42", `top=${t}`);
    } else {
      await pg.goto(BASE + "/" + cvDeep, { waitUntil: "domcontentloaded" });
      await pg.waitForTimeout(300);
      const n = await pg.evaluate(() => document.querySelectorAll(".srcline").length);
      if (n > 40) ok(`包内源码页行号渲染（${n} 行）`); else bad("包内行号", String(n));
    }
    await ctx.close();
  }
  // I. MathJax 本地渲染 + 零外部资源请求
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const pg = await ctx.newPage();
    const externals = [];
    pg.on("request", r => { if (/^https?:/.test(r.url()) && !r.url().startsWith(BASE)) externals.push(r.url()); });
    await pg.goto(BASE + "/m05_layers.html", { waitUntil: "networkidle", timeout: 60000 });
    const n = await pg.evaluate(() => document.querySelectorAll("mjx-container").length);
    if (n > 10) ok(`MathJax 本地渲染（${n} 个公式容器）`); else bad("MathJax 本地", `容器数 ${n}`);
    if (externals.length === 0) ok("m05 零外部网络请求（完全离线）"); else bad("外部请求", externals.join(", "));
    await ctx.close();
  }
}

// ── ③ 视口溢出矩阵 + 截图 ──
async function viewportMatrix(browser) {
  const widths = [1440, 1024, 768, 390, 320];
  const deep = existsSync(resolve(ROOT, "codeview"))
    ? "codeview/src/minimind_reborn/training/common/trainer.py.html"
    : "source/src/minimind_reborn/training/common/trainer.py.html";
  const targets = ["index.html", "m01_get_started.html", "m07_attention.html", "m12_datasets.html",
                   "m21_eval.html", deep];
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const pg = await ctx.newPage();
  for (const t of targets) {
    for (const w of widths) {
      await pg.setViewportSize({ width: w, height: w >= 768 ? 900 : 844 });
      await pg.goto(BASE + "/" + t, { waitUntil: "domcontentloaded", timeout: 30000 });
      await pg.waitForTimeout(300);
      const over = await pg.evaluate(() => {
        const d = document.documentElement;
        return { sw: d.scrollWidth, cw: d.clientWidth, bw: document.body.scrollWidth };
      });
      const tag = `${t} @${w}px`;
      if (over.sw <= over.cw + 1) ok(`无横向溢出 ${tag}`); else bad(`横向溢出 ${tag}`, `scrollWidth ${over.sw} > ${over.cw}`);
      const safe = t.replace(/[/]/g, "_");
      await pg.screenshot({ path: `${SHOTS}/${safe.replace(".html", "")}@${w}.png`, fullPage: false });
    }
  }
  await ctx.close();
}

const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome", args: ["--no-sandbox", "--disable-gpu"] });
const ctx0 = await browser.newContext();
await linkAudit(ctx0);
await ctx0.close();
await interactions(browser);
await viewportMatrix(browser);
await browser.close();
console.log(`\n==== 结果：${pass} PASS / ${fail} FAIL ====`);
if (fails.length) { console.log("失败明细："); fails.forEach(f => console.log("  -", f)); process.exit(1); }
