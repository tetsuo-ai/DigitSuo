#include <stdlib.h>
#include <ncurses.h>
#include <stdio.h>
#include "draw_interface.h"

extern FILE *debug_log;

static int is_valid_position(int x, int y)
{
    return (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE);
}

static void reset_cells(DrawGrid *grid)
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            grid->cells[i][j] = 0;
        }
    }
}

DrawGrid *init_grid(void)
{
    DrawGrid *grid = malloc(sizeof(DrawGrid));
    if (!grid)
        return NULL;
    reset_cells(grid);
    grid->cursor_x = GRID_SIZE / 2;
    grid->cursor_y = GRID_SIZE / 2;
    return grid;
}

void free_grid(DrawGrid *grid)
{
    free(grid);
}

static void clear_interface_lines(void)
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        move(i, 0);
        clrtoeol();
    }
}

static void draw_cells(const DrawGrid *grid)
{
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            int screen_y = i;
            int screen_x = j * CELL_WIDTH;
            if (grid->cells[i][j])
            {
                attron(A_BOLD);
                mvaddstr(screen_y, screen_x, "##");
                attroff(A_BOLD);
            }
            else
            {
                mvaddstr(screen_y, screen_x, ". ");
            }
        }
    }
}

void draw_interface(DrawGrid *grid)
{
    clear_interface_lines();
    draw_cells(grid);
}

static void draw_point_with_neighbors(DrawGrid *grid, int x, int y)
{
    if (is_valid_position(x, y))
        grid->cells[y][x] = 1;
    if (is_valid_position(x + 1, y))
        grid->cells[y][x + 1] = 1;
    if (is_valid_position(x - 1, y))
        grid->cells[y][x - 1] = 1;
    if (is_valid_position(x, y + 1))
        grid->cells[y + 1][x] = 1;
    if (is_valid_position(x, y - 1))
        grid->cells[y - 1][x] = 1;
}

static void draw_line(DrawGrid *grid, int x0, int y0, int x1, int y1)
{
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2;
    int e2;
    while (1)
    {
        draw_point_with_neighbors(grid, x0, y0);
        if (x0 == x1 && y0 == y1)
            break;
        e2 = err;
        if (e2 > -dx)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dy)
        {
            err += dx;
            y0 += sy;
        }
    }
}

void handle_mouse_event(DrawGrid *grid, int x, int y)
{
    int grid_x = x / CELL_WIDTH;
    int grid_y = y;
    if (!is_valid_position(grid_x, grid_y))
    {
        if (debug_log)
        {
            fprintf(debug_log, "Mouse event out of bounds: (%d,%d) -> grid(%d,%d)\n", x, y, grid_x, grid_y);
            fflush(debug_log);
        }
        return;
    }
    if (debug_log)
    {
        fprintf(debug_log, "Drawing at grid position: (%d,%d)\n", grid_x, grid_y);
        fflush(debug_log);
    }
    int prev_x = grid->cursor_x;
    int prev_y = grid->cursor_y;
    grid->cursor_x = grid_x;
    grid->cursor_y = grid_y;
    draw_line(grid, prev_x, prev_y, grid_x, grid_y);
}

void move_cursor(DrawGrid *grid, int direction)
{
    switch (direction)
    {
    case KEY_UP:
        if (grid->cursor_y > 0)
            grid->cursor_y--;
        break;
    case KEY_DOWN:
        if (grid->cursor_y < GRID_SIZE - 1)
            grid->cursor_y++;
        break;
    case KEY_LEFT:
        if (grid->cursor_x > 0)
            grid->cursor_x--;
        break;
    case KEY_RIGHT:
        if (grid->cursor_x < GRID_SIZE - 1)
            grid->cursor_x++;
        break;
    }
}

void toggle_cell(DrawGrid *grid)
{
    grid->cells[grid->cursor_y][grid->cursor_x] = !grid->cells[grid->cursor_y][grid->cursor_x];
}

void clear_grid(DrawGrid *grid)
{
    reset_cells(grid);
}
