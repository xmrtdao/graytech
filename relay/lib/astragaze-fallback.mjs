/**
 * The AstraGaze fallback page.
 *
 * WHY THIS EXISTS. graytech.mobilemonero.com is the live console. It loads
 * insightface and ONNX models on boot, which takes about 70 seconds on this
 * machine, and its measured outage was 56-92 seconds per restart before the
 * supervisor was fixed. During that window the product's front door is shut.
 *
 * A public demo that vanishes during a routine restart is not a service. So the
 * marketing surface has to live somewhere the face service cannot take down with
 * it. Cloudflare Pages was the first choice and is blocked: the token in
 * relay/.env verifies and reads DNS zones but 403s at account scope
 * (code=10000), and the one in
 * Desktop/31Harbor-Master/comms/Resend and Cloudflare CF Account API Keys.txt has
 * account access but no Pages permission AND belongs to a different account
 * (Hamptons31harbor, which does not serve mobilemonero.com).
 *
 * So: the relay. A separate process, no dependency on the recognition model,
 * already public through the same tunnel, and no new credentials needed.
 *
 * WHAT THE PAGE IS NOT. It is not a mirror of the console and does not pretend to
 * be. It says plainly, first thing, that the live console is restarting. A
 * fallback that looks like the working product is worse than an outage, because
 * a visitor concludes the system is broken rather than unavailable.
 *
 * Design tokens are lifted from graytech/graytech/server.py's :root block so the
 * two pages are visibly the same brand. Change them there and here together.
 *
 * ---------------------------------------------------------------------------
 * TWO EDITS LIVE IN relay/server.js, WHICH IS GITIGNORED (.gitignore line 34).
 * Recorded here because nothing else would tell you they exist, and a restored
 * or rebuilt server.js loses both silently.
 *
 * 1. The mount, next to the other static routes:
 *
 *      try {
 *        const { mountAstraGazeFallback } = await import('./lib/astragaze-fallback.mjs');
 *        mountAstraGazeFallback(app);
 *      } catch (e) {
 *        console.error('[astragaze] fallback page failed to mount:', e.message);
 *      }
 *
 * 2. The auth allowlist, in the middleware that guards every tunnel request:
 *
 *      req.path === '/graytech' || req.path === '/astragaze' ||
 *      req.path.startsWith('/graytech-assets/') ||
 *
 * WITHOUT (2) the page is 401 to the public. That middleware treats any request
 * carrying cf-ray or cf-connecting-ip as external and demands credentials, so a
 * brochure whose entire purpose is to be readable during an outage was itself
 * unreachable - which is the same class of failure as serving the brochure from
 * the service that goes down. The allowlist is the only difference between
 * "reachable" and "not", and it is one line in a file git does not track.
 *
 * Verify after any server.js change, from OUTSIDE the machine:
 *   curl -sI https://astragaze.mobilemonero.com/graytech | head -1
 *   -> HTTP/2 200, with x-astragaze-fallback: true
 * ---------------------------------------------------------------------------
 */

import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const PAGE_DIR = join(HERE, '..', 'public', 'graytech-offline');
export const PAGE = join(PAGE_DIR, 'index.html');

/**
 * Mount the fallback routes on an express app.
 *
 * @param {import('express').Express} app
 */
export function mountAstraGazeFallback(app) {
  // Two entry points, one file. Someone who reaches the console by typing the
  // product name should land on a page that explains what the product is, and
  // someone following a link from the console should get the same thing. Two
  // routes rather than two copies, which would drift.
  for (const route of ['/graytech', '/astragaze']) {
    app.get(route, (req, res) => {
      if (!existsSync(PAGE)) {
        // Say why instead of serving a bare 404. This page's entire job is to
        // explain a failure, so it must not fail quietly itself.
        return res
          .status(503)
          .type('text/plain')
          .send(
            'AstraGaze fallback page is missing from disk.\n' +
              `expected: ${PAGE}\n` +
              'The recognition service is unaffected - try graytech.mobilemonero.com\n'
          );
      }
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      // Short cache. This is the page someone reads while waiting out a restart,
      // so a stale copy either describes an outage that has ended or misses one
      // that has started. Cloudflare revalidates on no-cache.
      res.setHeader('Cache-Control', 'no-cache, must-revalidate');
      // Lets a test assert it came from here and not from the face service,
      // which is the whole point: if the header is absent, the fallback is dead.
      res.setHeader('X-AstraGaze-Fallback', 'true');
      res.sendFile(PAGE);
    });
  }

  // Assets, if the page grows any. One file today, named explicitly: a path
  // parameter joined onto a directory and read is a traversal, and '..' reaches
  // relay/.env. Same allowlist discipline as the relay's image routes.
  const ALLOWED = new Set(['index.html', 'styles.css', 'astragaze.js']);
  app.get('/graytech-assets/:name', (req, res) => {
    if (!ALLOWED.has(req.params.name)) {
      return res.status(404).send('/* asset not found */');
    }
    const filePath = join(PAGE_DIR, req.params.name);
    if (!existsSync(filePath)) return res.status(404).send('/* asset not found */');
    res.setHeader('Cache-Control', 'no-cache, must-revalidate');
    res.sendFile(filePath);
  });
}
