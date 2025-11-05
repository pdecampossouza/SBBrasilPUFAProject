from pylatex import Document, NoEscape

doc = Document()
doc.append(NoEscape(r"\input{examples/cnn_model/cnn_model.tex}"))
doc.generate_pdf("cnn_model", clean_tex=False)
