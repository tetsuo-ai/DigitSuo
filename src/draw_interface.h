#ifndef DRAW_INTERFACE_H
#define DRAW_INTERFACE_H

#define GRID_SIZE 28
#define CELL_WIDTH 2

typedef struct
{
    int cells[GRID_SIZE][GRID_SIZE];
    int cursor_x;
    int cursor_y;
} DrawGrid;

DrawGrid *init_grid(void);
void free_grid(DrawGrid *grid);
void draw_interface(DrawGrid *grid);
void move_cursor(DrawGrid *grid, int direction);
void toggle_cell(DrawGrid *grid);
void clear_grid(DrawGrid *grid);
void handle_mouse_event(DrawGrid *grid, int x, int y);

#endif // DRAW_INTERFACE_H
