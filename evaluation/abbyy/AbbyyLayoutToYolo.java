// AbbyyLayoutToYolo.java
//
// Runs ABBYY FineReader Engine 12 (Linux, Java API) layout analysis over a
// folder of page images and writes YOLO-format label files in OUR 4-class
// space, so the output scores with run_literature_eval.py exactly like Google
// Document AI / Azure DI (a single, un-thresholdable operating point -- FRE
// gives no per-block confidence, so no score column is written).
//
// Class map (our schema): 0 header, 1 text-area, 2 footnote, 3 footer.
//
// Block -> class decision (see mapBlock):
//   * Text block whose paragraph role is PR_Footnote / PR_Endnote  -> 2 footnote
//   * Running title (ITextBlock.BlockRole == BR_RunningTitle, or a
//     PR_RunningTitle paragraph)  -> 0 header if the block's vertical centre is
//     in the top half of the page, else 3 footer
//   * Any other text / table / picture block  -> 1 text-area (body)
//   * Separators, barcodes, checkmarks  -> dropped
// FRE has no first-class header/footer/footnote *block* type, so header vs
// footer is disambiguated by page position; footnote relies on the paragraph
// role assigned during document synthesis (needs Process, not just Analyze).
//
// Coordinates come from IRegion (pixels on the deskewed B/W plane); we take the
// union bounding rectangle of the block region and normalise by the layout
// width/height. Deskew shifts are small and our evaluator is AABB-tolerant, so
// no CoordinatesConverter round-trip to the original plane is done here.
//
// Build + run: see run_abbyy.sh in this folder.
//
// NOTE: a couple of Java-wrapper getter names differ subtly between FRE builds.
// The ones used here (getCount/getElement on collections; getLeft/getTop/
// getRight/getBottom(index) on IRegion; getType/getBlockRole/getText/
// getParagraphs/getParagraphStyle/getParagraphRole) match the FRE 12 Java
// wrapper and the shipped Samples/Hello project. If your build differs, check
// <FRE>/Samples/Java and <FRE>/Inc/Java and adjust.

import com.abbyy.FREngine.*;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class AbbyyLayoutToYolo {

    // our class ids
    static final int HEADER = 0, TEXT_AREA = 1, FOOTNOTE = 2, FOOTER = 3;

    static final String[] IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"};

    static IEngine engine = null;

    public static void main(String[] args) {
        if (args.length < 6) {
            System.err.println("usage: AbbyyLayoutToYolo <dllFolder> <customerProjectId> "
                    + "<licensePath> <licensePassword> <imageDir> <outDir> [maxPages]");
            System.exit(2);
        }
        String dllFolder = args[0];
        String customerProjectId = args[1];
        String licensePath = args[2];
        String licensePassword = args[3];
        File imageDir = new File(args[4]);
        File outDir = new File(args[5]);
        int maxPages = args.length > 6 ? Integer.parseInt(args[6]) : 0;

        File lblDir = new File(outDir, "labels");
        lblDir.mkdirs();

        try {
            // FRE 12 Linux: only InitializeEngine is available (no IEngineLoader).
            engine = Engine.InitializeEngine(dllFolder, customerProjectId,
                    licensePath, licensePassword, "", "", false);
            System.out.println("Engine loaded.");
            try {
                // Full layout profile; DocumentConversion keeps text/table/picture
                // blocks and runs synthesis so paragraph roles (footnote/running
                // title) are populated.
                engine.LoadPredefinedProfile("DocumentConversion_Accuracy");

                List<File> imgs = listImages(imageDir);
                System.out.println(imgs.size() + " images -> " + lblDir);
                int n = 0, ok = 0, fail = 0;
                for (File ip : imgs) {
                    if (maxPages > 0 && n >= maxPages) break;
                    n++;
                    File dst = new File(lblDir, stem(ip) + ".txt");
                    if (dst.exists()) { ok++; continue; }
                    try {
                        List<String> lines = processOne(ip);
                        Files.write(dst.toPath(),
                                String.join("\n", lines).getBytes(StandardCharsets.UTF_8));
                        ok++;
                    } catch (Exception ex) {
                        fail++;
                        System.out.println("!! " + ip.getName() + ": " + ex.getMessage());
                    }
                    if (n % 50 == 0) System.out.println("  " + n + "/" + imgs.size() + " ...");
                }
                // names file for the scorer
                Files.write(new File(outDir, "data.yaml").toPath(),
                        ("names:\n  0: header\n  1: text-area\n  2: footnote\n  3: footer\n")
                                .getBytes(StandardCharsets.UTF_8));
                System.out.println("done: " + ok + " ok, " + fail + " failed -> " + lblDir);
            } finally {
                engine = null;
                System.runFinalization();
                Engine.DeinitializeEngine();
            }
        } catch (Exception ex) {
            System.err.println("FATAL: " + ex.getMessage());
            ex.printStackTrace();
            System.exit(1);
        }
    }

    static List<String> processOne(File ip) throws Exception {
        IFRDocument document = engine.CreateFRDocument();
        List<String> lines = new ArrayList<>();
        try {
            document.AddImageFile(ip.getAbsolutePath(), null, null);
            // Full processing: analysis + recognition + synthesis. Synthesis is
            // what assigns block/paragraph roles. (Recognition text is unused;
            // Tibetan is not an FRE language, but geometry + roles come from the
            // layout model, not from character recognition.)
            document.Process(null);

            IFRPages pages = document.getPages();
            if (pages.getCount() == 0) return lines;
            IFRPage page = pages.getElement(0);
            ILayout layout = page.getLayout();
            int W = layout.getWidth();
            int H = layout.getHeight();
            if (W <= 0 || H <= 0) return lines;

            ILayoutBlocks blocks = layout.getBlocks();
            for (int i = 0; i < blocks.getCount(); i++) {
                IBlock block = blocks.getElement(i);
                int[] bb = regionBounds(block.getRegion());   // {l,t,r,b} or null
                if (bb == null) continue;
                int cls = mapBlock(block, bb, H);
                if (cls < 0) continue;
                lines.add(toYolo(cls, bb, W, H));
            }
        } finally {
            document.Close();
        }
        return lines;
    }

    // Returns our class id, or -1 to drop the block.
    static int mapBlock(IBlock block, int[] bb, int pageH) throws Exception {
        BlockTypeEnum type = block.getType();
        if (type == BlockTypeEnum.BT_Separator || type == BlockTypeEnum.BT_SeparatorsGroup
                || type == BlockTypeEnum.BT_Barcode || type == BlockTypeEnum.BT_Checkmark
                || type == BlockTypeEnum.BT_CheckmarkGroup) {
            return -1;
        }
        if (type != BlockTypeEnum.BT_Text) {
            // table / raster picture / vector picture -> body content
            return TEXT_AREA;
        }

        ITextBlock tb = block.GetAsTextBlock();
        boolean isFootnote = false;
        boolean isRunning = (tb.getBlockRole() == BlockRoleEnum.BR_RunningTitle);
        try {
            IParagraphs pars = tb.getText().getParagraphs();
            for (int p = 0; p < pars.getCount(); p++) {
                ParagraphRoleEnum role = pars.getElement(p).getParagraphStyle().getParagraphRole();
                if (role == ParagraphRoleEnum.PR_Footnote || role == ParagraphRoleEnum.PR_Endnote) {
                    isFootnote = true;
                } else if (role == ParagraphRoleEnum.PR_RunningTitle) {
                    isRunning = true;
                }
            }
        } catch (Exception ignore) {
            // paragraph roles unavailable (e.g. Analyze-only) -> fall back to block role
        }

        if (isFootnote) return FOOTNOTE;
        if (isRunning) {
            int cy = (bb[1] + bb[3]) / 2;
            return (cy < pageH / 2) ? HEADER : FOOTER;
        }
        return TEXT_AREA;
    }

    // Union bounding rectangle of an IRegion, or null if empty.
    static int[] regionBounds(IRegion region) throws Exception {
        int count = region.getCount();
        if (count == 0) return null;
        int l = Integer.MAX_VALUE, t = Integer.MAX_VALUE, r = Integer.MIN_VALUE, b = Integer.MIN_VALUE;
        for (int i = 0; i < count; i++) {
            l = Math.min(l, region.getLeft(i));
            t = Math.min(t, region.getTop(i));
            r = Math.max(r, region.getRight(i));
            b = Math.max(b, region.getBottom(i));
        }
        if (r <= l || b <= t) return null;
        return new int[]{l, t, r, b};
    }

    static String toYolo(int cls, int[] bb, int W, int H) {
        double cx = clamp(((bb[0] + bb[2]) / 2.0) / W);
        double cy = clamp(((bb[1] + bb[3]) / 2.0) / H);
        double w = clamp((bb[2] - bb[0]) / (double) W);
        double h = clamp((bb[3] - bb[1]) / (double) H);
        return String.format(Locale.US, "%d %.6f %.6f %.6f %.6f", cls, cx, cy, w, h);
    }

    static double clamp(double v) { return Math.max(0.0, Math.min(1.0, v)); }

    static List<File> listImages(File dir) {
        File[] all = dir.listFiles();
        List<File> out = new ArrayList<>();
        if (all == null) return out;
        for (File f : all) {
            String name = f.getName().toLowerCase(Locale.US);
            for (String ext : IMG_EXTS) {
                if (name.endsWith(ext)) { out.add(f); break; }
            }
        }
        out.sort((a, c) -> a.getName().compareTo(c.getName()));
        return out;
    }

    static String stem(File f) {
        String n = f.getName();
        int dot = n.lastIndexOf('.');
        return dot > 0 ? n.substring(0, dot) : n;
    }
}
