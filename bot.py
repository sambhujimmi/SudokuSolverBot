from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import cv2
import numpy as np
import pytesseract
import easyocr
import dotenv
import os
from sudoku_solver import solver  # Import your Sudoku solver
from extract_sudoku import process_sudoku

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Replace with your bot token
# BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
os.environ.clear()
dotenv.load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("You must set the BOT_TOKEN environment variable")

reader = easyocr.Reader(['en'])

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Send me a Sudoku image, and I'll solve it!")

async def process_image(update: Update, context: CallbackContext):
    file = await update.message.photo[-1].get_file()
    await file.download_to_drive("sudoku.jpg")

    # Extract Sudoku grid
    grid = extract_sudoku("sudoku.jpg")

    if grid:
        solved_grid = solver(grid)  # Use your existing solver
        solution_text = "\n".join([" ".join(map(str, row)) for row in solved_grid])
        await update.message.reply_text(f"Here is the solved Sudoku:\n{solution_text}")
    else:
        await update.message.reply_text("Could not recognize the Sudoku. Please try again!")

def extract_sudoku(image_path):
    """Extract Sudoku grid using OpenCV + Tesseract OCR"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    sudoku_grid = []
    height, width = img.shape
    cell_h, cell_w = height // 9, width // 9

    for i in range(9):
        row = []
        for j in range(9):
            cell = img[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            text = reader.readtext(cell, detail=0, allowlist="123456789")
            
            try:
                digit = text[0]
            except:
                digit = 0
            row.append(int(digit))
        sudoku_grid.append(row)

    return sudoku_grid

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, process_image))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()