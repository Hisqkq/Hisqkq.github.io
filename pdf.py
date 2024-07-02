from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Le Chat Curieux', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body.encode('latin-1', 'replace').decode('latin-1'))
        self.ln()

pdf = PDF()
pdf.add_page()
pdf.chapter_title('The Curious Cat')
# Ici on peut ajouter un texte
story = """
Once upon a time, there was a little cat named Minou who lived in a charming country house. Minou was known for his endless curiosity. Every day, he explored every nook and cranny of the house, looking for new adventures.
One morning, as the sun was just rising, Minou spotted a small, sparkling light under the old oak tree in the garden. Intrigued, he approached slowly and discovered a firefly trapped in a spider's web.
With great care, Minou freed the firefly. "Thank you, kind cat," said the firefly. "To reward you, I will take you to the heart of the enchanted forest where you will find unimaginable wonders."
Curious and excited, Minou followed the firefly through the garden and beyond, until he entered the forest. There, he discovered a fantastic world where the trees whispered ancient secrets, the flowers sang enchanting melodies, and the animals spoke a mysterious language.
Minou spent the day exploring this wonderful world. He met a wise fox who taught him the art of discretion, an agile squirrel who showed him how to climb the tallest trees, and a learned owl who shared ancient stories with him.
As night fell, Minou returned home, his heart filled with magical memories. Every day, he dreamed of returning to the enchanted forest, and every night, he snuggled into his basket, grateful for the day's adventure.
And so, Minou, the curious cat, continued to have extraordinary adventures, proving that curiosity is not a flaw, but a door to unexpected wonders.
The end.
"""
pdf.chapter_body(story)

# Sauvegarder le PDF
output_path = "the_curious_cat.pdf"
pdf.output(output_path)
output_path