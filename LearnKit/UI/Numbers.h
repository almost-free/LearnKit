/*
 * Numbers.h
 */

#import <AppKit/AppKit.h>
#import <ScriptingBridge/ScriptingBridge.h>


@class NumbersApplication, NumbersDocument, NumbersWindow, NumbersRichText, NumbersCharacter, NumbersParagraph, NumbersWord, NumbersPlaceholderText, NumbersIWorkContainer, NumbersIWorkItem, NumbersAudioClip, NumbersShape, NumbersChart, NumbersImage, NumbersGroup, NumbersLine, NumbersMovie, NumbersTable, NumbersTextItem, NumbersRange, NumbersCell, NumbersRow, NumbersColumn, NumbersSheet, NumbersTemplate;

enum NumbersSaveOptions {
	NumbersSaveOptionsYes = 'yes ' /* Save the file. */,
	NumbersSaveOptionsNo = 'no  ' /* Do not save the file. */,
	NumbersSaveOptionsAsk = 'ask ' /* Ask the user whether or not to save the file. */
};
typedef enum NumbersSaveOptions NumbersSaveOptions;

enum NumbersPrintingErrorHandling {
	NumbersPrintingErrorHandlingStandard = 'lwst' /* Standard PostScript error handling */,
	NumbersPrintingErrorHandlingDetailed = 'lwdt' /* print a detailed report of PostScript errors */
};
typedef enum NumbersPrintingErrorHandling NumbersPrintingErrorHandling;

enum NumbersTAVT {
	NumbersTAVTBottom = 'avbt' /* Right-align content. */,
	NumbersTAVTCenter = 'actr' /* Center-align content. */,
	NumbersTAVTTop = 'avtp' /* Top-align content. */
};
typedef enum NumbersTAVT NumbersTAVT;

enum NumbersTAHT {
	NumbersTAHTAutoAlign = 'aaut' /* Auto-align based on content type. */,
	NumbersTAHTCenter = 'actr' /* Center-align content. */,
	NumbersTAHTJustify = 'ajst' /* Fully justify (left and right) content. */,
	NumbersTAHTLeft = 'alft' /* Left-align content. */,
	NumbersTAHTRight = 'arit' /* Right-align content. */
};
typedef enum NumbersTAHT NumbersTAHT;

enum NumbersNMSD {
	NumbersNMSDAscending = 'ascn' /* Sort in increasing value order */,
	NumbersNMSDDescending = 'dscn' /* Sort in decreasing value order */
};
typedef enum NumbersNMSD NumbersNMSD;

enum NumbersNMCT {
	NumbersNMCTAutomatic = 'faut' /* Automatic format */,
	NumbersNMCTCheckbox = 'fcch' /* Checkbox control format (Numbers only) */,
	NumbersNMCTCurrency = 'fcur' /* Currency number format */,
	NumbersNMCTDateAndTime = 'fdtm' /* Date and time format */,
	NumbersNMCTFraction = 'ffra' /* Fraction number format */,
	NumbersNMCTNumber = 'nmbr' /* Decimal number format */,
	NumbersNMCTPercent = 'fper' /* Percentage number format */,
	NumbersNMCTPopUpMenu = 'fcpp' /* Pop-up menu control format (Numbers only) */,
	NumbersNMCTScientific = 'fsci' /* Scientific notation format */,
	NumbersNMCTSlider = 'fcsl' /* Slider control format (Numbers only) */,
	NumbersNMCTStepper = 'fcst' /* Stepper control format (Numbers only) */,
	NumbersNMCTText = 'ctxt' /* Text format */,
	NumbersNMCTDuration = 'fdur' /* Duration format */,
	NumbersNMCTRating = 'frat' /* Rating format. (Numbers only) */,
	NumbersNMCTNumeralSystem = 'fcns' /* Numeral System */
};
typedef enum NumbersNMCT NumbersNMCT;

enum NumbersItemFillOptions {
	NumbersItemFillOptionsNoFill = 'fino' /*  */,
	NumbersItemFillOptionsColorFill = 'fico' /*  */,
	NumbersItemFillOptionsGradientFill = 'figr' /*  */,
	NumbersItemFillOptionsAdvancedGradientFill = 'fiag' /*  */,
	NumbersItemFillOptionsImageFill = 'fiim' /*  */,
	NumbersItemFillOptionsAdvancedImageFill = 'fiai' /*  */
};
typedef enum NumbersItemFillOptions NumbersItemFillOptions;

enum NumbersPlaybackRepetitionMethod {
	NumbersPlaybackRepetitionMethodNone = 'mvrn' /*  */,
	NumbersPlaybackRepetitionMethodLoop = 'mvlp' /*  */,
	NumbersPlaybackRepetitionMethodLoopBackAndForth = 'mvbf' /*  */
};
typedef enum NumbersPlaybackRepetitionMethod NumbersPlaybackRepetitionMethod;

enum NumbersSaveableFileFormat {
	NumbersSaveableFileFormatNumbers = 'Nuff' /* The Numbers native file format */
};
typedef enum NumbersSaveableFileFormat NumbersSaveableFileFormat;

enum NumbersExportFormat {
	NumbersExportFormatPDF = 'Npdf' /* PDF */,
	NumbersExportFormatMicrosoftExcel = 'Nexl' /* Microsoft Excel */,
	NumbersExportFormatCSV = 'Ncsv' /* CSV */,
	NumbersExportFormatNumbers09 = 'Nnmb' /* Numbers 09 */
};
typedef enum NumbersExportFormat NumbersExportFormat;



/*
 * Standard Suite
 */

// The application's top-level scripting object.
@interface NumbersApplication : SBApplication

- (SBElementArray *) documents;
- (SBElementArray *) windows;

@property (copy, readonly) NSString *name;  // The name of the application.
@property (readonly) BOOL frontmost;  // Is this the active application?
@property (copy, readonly) NSString *version;  // The version number of the application.

- (id) open:(id)x;  // Open a document.
- (void) print:(id)x withProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) quitSaving:(NumbersSaveOptions)saving;  // Quit the application.
- (BOOL) exists:(id)x;  // Verify that an object exists.

@end

// A document.
@interface NumbersDocument : SBObject

@property (copy, readonly) NSString *name;  // Its name.
@property (readonly) BOOL modified;  // Has it been modified since the last save?
@property (copy, readonly) NSURL *file;  // Its location on disk, if it has one.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.
- (void) exportTo:(NSURL *)to as:(NumbersExportFormat)as withProperties:(NSDictionary *)withProperties;  // Export a document to another file

@end

// A window.
@interface NumbersWindow : SBObject

@property (copy, readonly) NSString *name;  // The title of the window.
- (NSInteger) id;  // The unique identifier of the window.
@property NSInteger index;  // The index of the window, ordered front to back.
@property NSRect bounds;  // The bounding rectangle of the window.
@property (readonly) BOOL closeable;  // Does the window have a close button?
@property (readonly) BOOL miniaturizable;  // Does the window have a minimize button?
@property BOOL miniaturized;  // Is the window minimized right now?
@property (readonly) BOOL resizable;  // Can the window be resized?
@property BOOL visible;  // Is the window visible right now?
@property (readonly) BOOL zoomable;  // Does the window have a zoom button?
@property BOOL zoomed;  // Is the window zoomed right now?
@property (copy, readonly) NumbersDocument *document;  // The document whose contents are displayed in the window.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.

@end



/*
 * iWork Text Suite
 */

// This provides the base rich text class for all iWork applications.
@interface NumbersRichText : SBObject

- (SBElementArray *) characters;
- (SBElementArray *) paragraphs;
- (SBElementArray *) words;
- (SBElementArray *) placeholderTexts;

@property (copy) NSColor *color;  // The color of the font. Expressed as an RGB value consisting of a list of three color values from 0 to 65535. ex: Blue = {0, 0, 65535}.
@property (copy) NSString *font;  // The name of the font.  Can be the PostScript name, such as: “TimesNewRomanPS-ItalicMT”, or display name: “Times New Roman Italic”. TIP: Use the Font Book application get the information about a typeface.
@property NSInteger size;  // The size of the font.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.

@end

// One of some text's characters.
@interface NumbersCharacter : NumbersRichText


@end

// One of some text's paragraphs.
@interface NumbersParagraph : NumbersRichText

- (SBElementArray *) characters;
- (SBElementArray *) words;
- (SBElementArray *) placeholderTexts;


@end

// One of some text's words.
@interface NumbersWord : NumbersRichText

- (SBElementArray *) characters;


@end

// One of some text's placeholders.
@interface NumbersPlaceholderText : NumbersRichText

@property (copy) NSString *tag;  // Its script tag.


@end



/*
 * iWork Suite
 */

// A container for iWork items
@interface NumbersIWorkContainer : SBObject

- (SBElementArray *) audioClips;
- (SBElementArray *) charts;
- (SBElementArray *) images;
- (SBElementArray *) iWorkItems;
- (SBElementArray *) groups;
- (SBElementArray *) lines;
- (SBElementArray *) movies;
- (SBElementArray *) shapes;
- (SBElementArray *) tables;
- (SBElementArray *) textItems;

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.

@end

// An item which supports formatting
@interface NumbersIWorkItem : SBObject

@property NSInteger height;  // The height of the iWork item.
@property BOOL locked;  // Whether the object is locked.
@property (copy, readonly) NumbersIWorkContainer *parent;  // The iWork container containing this iWork item.
@property NSPoint position;  // The horizontal and vertical coordinates of the top left point of the iWork item.
@property NSInteger width;  // The width of the iWork item.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.

@end

// An audio clip
@interface NumbersAudioClip : NumbersIWorkItem

@property (copy) id fileName;  // The name of the audio file.
@property NSInteger clipVolume;  // The volume setting for the audio clip, from 0 (none) to 100 (full volume).
@property NumbersPlaybackRepetitionMethod repetitionMethod;  // If or how the audio clip repeats.


@end

// A shape container
@interface NumbersShape : NumbersIWorkItem

@property (readonly) NumbersItemFillOptions backgroundFillType;  // The background, if any, for the shape.
@property (copy) NumbersRichText *objectText;  // The text contained within the shape.
@property BOOL reflectionShowing;  // Is the iWork item displaying a reflection?
@property NSInteger reflectionValue;  // The percentage of reflection of the iWork item, from 0 (none) to 100 (full).
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.
@property NSInteger opacity;  // The opacity of the object, in percent.


@end

// A chart
@interface NumbersChart : NumbersIWorkItem


@end

// An image container
@interface NumbersImage : NumbersIWorkItem

@property (copy) NSString *objectDescription;  // Text associated with the image, read aloud by VoiceOver.
@property (copy, readonly) NSURL *file;  // The image file.
@property (copy) id fileName;  // The name of the image file.
@property NSInteger opacity;  // The opacity of the object, in percent.
@property BOOL reflectionShowing;  // Is the iWork item displaying a reflection?
@property NSInteger reflectionValue;  // The percentage of reflection of the iWork item, from 0 (none) to 100 (full).
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.


@end

// A group container
@interface NumbersGroup : NumbersIWorkContainer

@property NSInteger height;  // The height of the iWork item.
@property (copy, readonly) NumbersIWorkContainer *parent;  // The iWork container containing this iWork item.
@property NSPoint position;  // The horizontal and vertical coordinates of the top left point of the iWork item.
@property NSInteger width;  // The width of the iWork item.
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.


@end

// A line
@interface NumbersLine : NumbersIWorkItem

@property NSPoint endPoint;  // A list of two numbers indicating the horizontal and vertical position of the line ending point.
@property BOOL reflectionShowing;  // Is the iWork item displaying a reflection?
@property NSInteger reflectionValue;  // The percentage of reflection of the iWork item, from 0 (none) to 100 (full).
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.
@property NSPoint startPoint;  // A list of two numbers indicating the horizontal and vertical position of the line starting point.


@end

// A movie container
@interface NumbersMovie : NumbersIWorkItem

@property (copy) id fileName;  // The name of the movie file.
@property NSInteger movieVolume;  // The volume setting for the movie, from 0 (none) to 100 (full volume).
@property NSInteger opacity;  // The opacity of the object, in percent.
@property BOOL reflectionShowing;  // Is the iWork item displaying a reflection?
@property NSInteger reflectionValue;  // The percentage of reflection of the iWork item, from 0 (none) to 100 (full).
@property NumbersPlaybackRepetitionMethod repetitionMethod;  // If or how the movie repeats.
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.


@end

// A table
@interface NumbersTable : NumbersIWorkItem

- (SBElementArray *) cells;
- (SBElementArray *) rows;
- (SBElementArray *) columns;
- (SBElementArray *) ranges;

@property (copy) NSString *name;  // The item's name.
@property (copy, readonly) NumbersRange *cellRange;  // The range describing every cell in the table.
@property (copy) NumbersRange *selectionRange;  // The cells currently selected in the table.
@property NSInteger rowCount;  // The number of rows in the table.
@property NSInteger columnCount;  // The number of columns in the table.
@property NSInteger headerRowCount;  // The number of header rows in the table.
@property NSInteger headerColumnCount;  // The number of header columns in the table.
@property NSInteger footerRowCount;  // The number of footer rows in the table.

- (void) sortBy:(NumbersColumn *)by direction:(NumbersNMSD)direction inRows:(NumbersRange *)inRows;  // Sort the rows of the table.
- (void) transpose;  // Transpose the rows and columns of the table.

@end

// A text container
@interface NumbersTextItem : NumbersIWorkItem

@property (readonly) NumbersItemFillOptions backgroundFillType;  // The background, if any, for the text item.
@property (copy) NumbersRichText *objectText;  // The text contained within the text item.
@property NSInteger opacity;  // The opacity of the object, in percent.
@property BOOL reflectionShowing;  // Is the iWork item displaying a reflection?
@property NSInteger reflectionValue;  // The percentage of reflection of the iWork item, from 0 (none) to 100 (full).
@property NSInteger rotation;  // The rotation of the iWork item, in degrees from 0 to 359.


@end

// A range of cells in a table
@interface NumbersRange : SBObject

- (SBElementArray *) cells;
- (SBElementArray *) columns;
- (SBElementArray *) rows;

@property (copy) NSString *fontName;  // The font of the range's cells.
@property double fontSize;  // The font size of the range's cells.
@property NumbersNMCT format;  // The format of the range's cells.
@property NumbersTAHT alignment;  // The horizontal alignment of content in the range's cells.
@property (copy, readonly) NSString *name;  // The range's coordinates.
@property (copy) NSColor *textColor;  // The text color of the range's cells.
@property BOOL textWrap;  // Whether text should wrap in the range's cells.
@property (copy) NSColor *backgroundColor;  // The background color of the range's cells.
@property NumbersTAVT verticalAlignment;  // The vertical alignment of content in the range's cells.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.
- (void) clear;  // Clear the contents of a specified range of cells. Only content is removed; formatting and style are preserved.
- (void) merge;  // Merge a specified range of cells.
- (void) unmerge;  // Unmerge all merged cells in a specified range.
- (SBObject *) addColumnAfter;  // Add a column to the table after a specified range of cells.
- (SBObject *) addColumnBefore;  // Add a column to the table before a specified range of cells.
- (SBObject *) addRowAbove;  // Add a row to the table below a specified range of cells.
- (SBObject *) addRowBelow;  // Add a row to the table below a specified range of cells.
- (void) remove;  // Remove specified rows or columns from a table.

@end

// A cell in a table
@interface NumbersCell : NumbersRange

@property (copy, readonly) NumbersColumn *column;  // The cell's column.
@property (copy, readonly) NumbersRow *row;  // The cell's row.
@property (copy) id value;  // The actual value in the cell, or missing value if the cell is empty.
@property (copy, readonly) NSString *formattedValue;  // The formatted value in the cell, or missing value if the cell is empty.
@property (copy, readonly) NSString *formula;  // The formula in the cell, as text, e.g. =SUM(40+2). If the cell does not contain a formula, returns missing value. To set the value of a cell to a formula as text, use the value property.


@end

// A row of cells in a table
@interface NumbersRow : NumbersRange

@property (readonly) NSInteger address;  // The row's index in the table (e.g., the second row has address 2).
@property double height;  // The height of the row.


@end

// A column of cells in a table
@interface NumbersColumn : NumbersRange

@property (readonly) NSInteger address;  // The column's index in the table (e.g., the second column has address 2).
@property double width;  // The width of the column.


@end



/*
 * Numbers Suite
 */

// The Numbers application.
@interface NumbersApplication (NumbersSuite)

- (SBElementArray *) templates;

@end

// The Numbers document.
@interface NumbersDocument (NumbersSuite)

- (SBElementArray *) sheets;

- (NSString *) id;  // Document ID.
@property (copy, readonly) NumbersTemplate *documentTemplate;  // The template assigned to the document.
@property (copy) NumbersSheet *activeSheet;  // The active sheet.

@end

// A sheet in a document
@interface NumbersSheet : NumbersIWorkContainer

@property (copy) NSString *name;  // The sheet's name.


@end

@interface NumbersTable (NumbersSuite)

@property BOOL filtered;  // Whether the table is currently filtered.
@property BOOL headerRowsFrozen;  // Whether header rows are frozen.
@property BOOL headerColumnsFrozen;  // Whether header columns are frozen.

@end

// A styled document layout.
@interface NumbersTemplate : SBObject

- (NSString *) id;  // The identifier used by the application.
@property (copy, readonly) NSString *name;  // The localized name displayed to the user.

- (void) closeSaving:(NumbersSaveOptions)saving savingIn:(NSURL *)savingIn;  // Close a document.
- (void) saveIn:(NSURL *)in_ as:(NumbersSaveableFileFormat)as;  // Save a document.
- (void) printWithProperties:(NSDictionary *)withProperties printDialog:(BOOL)printDialog;  // Print a document.
- (void) delete;  // Delete an object.
- (void) duplicateTo:(SBObject *)to withProperties:(NSDictionary *)withProperties;  // Copy an object.
- (void) moveTo:(SBObject *)to;  // Move an object to a new location.

@end



/*
 * Compatibility Suite
 */

@interface NumbersRange (CompatibilitySuite)

@end

@interface NumbersRow (CompatibilitySuite)

@end

@interface NumbersColumn (CompatibilitySuite)

@end

