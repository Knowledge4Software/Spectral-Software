// Dump an authoritative METHOD-node -> source-file mapping for one CPG.
//
// joern-export writes one DOT per method and names the graph after the method,
// never after the file it came from.  Recovering "which submission does this
// DOT belong to" from filenames or export order is therefore guesswork: the C
// frontend emits ~5x more DOTs than source files (one <global> per file plus a
// stub per external <operator>), so positional matching silently desynchronises.
//
// This map is read by build_dot_index_with_method_map(), which looks up each
// DOT's root METHOD node id and recovers the exact source file.
//
// Parameters arrive as environment variables because the Joern CLI's --param
// handling differs between releases.
val cpgFile = sys.env("SPECTRAL_METHOD_MAP_CPG")
val outFile = sys.env("SPECTRAL_METHOD_MAP_OUT")

importCpg(cpgFile)

val rows = cpg.method.l.map { method =>
  s"${method.id}\t${method.filename}\t${method.fullName}"
}

java.nio.file.Files.write(
  java.nio.file.Paths.get(outFile),
  rows.mkString("\n").getBytes("UTF-8"),
)

println(s"[method-map] wrote ${rows.size} methods to $outFile")
