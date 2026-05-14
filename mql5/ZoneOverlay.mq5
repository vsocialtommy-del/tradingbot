//+------------------------------------------------------------------+
//|                                              ZoneOverlay.mq5    |
//|                              Bot zone visualization companion   |
//|                                                                  |
//|  Purpose:                                                        |
//|    Reads tradingbot_zones.csv (written by the Python bot every   |
//|    M5 close) and reconciles colored OBJ_RECTANGLE objects on     |
//|    the chart so the operator can see what zones the bot is      |
//|    currently tracking.                                          |
//|                                                                  |
//|  CSV format (PR #49):                                            |
//|    zone_id,direction,status,flipped_direction,top,bottom,        |
//|      formed_at_unix                                               |
//|                                                                  |
//|  Object naming:                                                  |
//|    BOTZONE_<zone_uuid>  — one rectangle per zone. Lets us        |
//|    reconcile (create/update/delete) without touching the         |
//|    operator's own chart objects.                                 |
//|                                                                  |
//|  Operator setup:                                                 |
//|    1. Compile this file in MetaEditor (F7).                      |
//|    2. Drag onto the XAUUSD M5 chart in the MT5 terminal.         |
//|    3. Allow "Algo Trading" in the popup.                         |
//|                                                                  |
//|  After that: rectangles appear within PollSeconds (default 5s)   |
//|  and refresh on every bot M5 close.                              |
//+------------------------------------------------------------------+

#property copyright   "Bot visualization companion"
#property description "Reads tradingbot_zones.csv and draws zones as rectangles."
#property version     "1.00"
#property strict

// Inputs the operator can tweak via MT5's EA properties dialog.
input string InpCsvFilename = "tradingbot_zones.csv";  // CSV the bot writes
input int    InpPollSeconds = 5;                       // poll interval (s)
input int    InpExtendHours = 4;                       // right edge: now + N hours
input bool   InpCleanupOnExit = true;                  // delete BOTZONE_* on EA removal

// Color scheme. clr* constants from MT5 stdlib.
input color  InpColorBuy        = clrSeaGreen;         // CONFIRMED BUY
input color  InpColorSell       = clrFireBrick;        // CONFIRMED SELL
input color  InpColorActiveBuy  = clrLimeGreen;        // ACTIVE BUY (open setup)
input color  InpColorActiveSell = clrCrimson;          // ACTIVE SELL (open setup)
input color  InpColorFlipped    = clrDarkOrange;       // FLIPPED (any direction)

// Object name prefix — lets us cleanly distinguish our objects from
// the operator's manual drawings.
const string OBJ_PREFIX = "BOTZONE_";


//+------------------------------------------------------------------+
//| Expert init                                                      |
//+------------------------------------------------------------------+
int OnInit()
{
    EventSetTimer(InpPollSeconds);
    Print("ZoneOverlay: initialized; polling '", InpCsvFilename,
          "' every ", InpPollSeconds, "s");
    // Trigger one immediate render so the operator sees zones right
    // away instead of waiting for the first timer fire.
    ReconcileFromFile();
    return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert deinit                                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    if(InpCleanupOnExit)
    {
        DeleteAllBotZones();
        Print("ZoneOverlay: cleaned up all BOTZONE_* objects on exit");
    }
}


//+------------------------------------------------------------------+
//| Timer — poll the CSV and reconcile                               |
//+------------------------------------------------------------------+
void OnTimer()
{
    ReconcileFromFile();
}


//+------------------------------------------------------------------+
//| Read CSV → create / update / delete rectangles                   |
//+------------------------------------------------------------------+
void ReconcileFromFile()
{
    if(!FileIsExist(InpCsvFilename))
    {
        // No file yet — bot hasn't written one. Quiet skip (logging
        // every poll would flood the Experts tab).
        return;
    }

    int fh = FileOpen(InpCsvFilename,
                      FILE_READ | FILE_CSV | FILE_ANSI | FILE_SHARE_READ,
                      ',');
    if(fh == INVALID_HANDLE)
    {
        Print("ZoneOverlay: FileOpen failed for '", InpCsvFilename,
              "', err=", GetLastError());
        return;
    }

    // Track which BOTZONE_* names appear in this snapshot — anything
    // missing on the next pass gets deleted.
    string seen_names[];
    ArrayResize(seen_names, 0);

    // Skip the header row.
    if(!FileIsEnding(fh))
    {
        for(int i = 0; i < 7; i++)
            FileReadString(fh);
    }

    int row_count = 0;
    while(!FileIsEnding(fh))
    {
        string zone_id           = FileReadString(fh);
        if(StringLen(zone_id) == 0) break;
        string direction         = FileReadString(fh);
        string status            = FileReadString(fh);
        string flipped_direction = FileReadString(fh);
        double top               = StringToDouble(FileReadString(fh));
        double bottom            = StringToDouble(FileReadString(fh));
        datetime formed_at       = (datetime)StringToInteger(
                                       FileReadString(fh));

        string obj_name = OBJ_PREFIX + zone_id;
        datetime right_edge = TimeCurrent() + InpExtendHours * 3600;
        color render_color = PickColor(direction, status, flipped_direction);

        if(ObjectFind(0, obj_name) < 0)
        {
            CreateZoneRectangle(obj_name, formed_at, top, right_edge,
                                bottom, render_color);
        }
        else
        {
            UpdateZoneRectangle(obj_name, right_edge, bottom, render_color);
        }

        int n = ArraySize(seen_names);
        ArrayResize(seen_names, n + 1);
        seen_names[n] = obj_name;
        row_count++;
    }
    FileClose(fh);

    // Cleanup pass: delete any BOTZONE_* not in this snapshot.
    DeleteOrphanZones(seen_names);

    if(row_count > 0)
    {
        // One-line summary per render. Quiet enough not to spam.
        // Comment out if even this is too noisy.
        // Print("ZoneOverlay: rendered ", row_count, " zone(s)");
    }
}


//+------------------------------------------------------------------+
//| ObjectCreate wrapper — full property setup                       |
//+------------------------------------------------------------------+
void CreateZoneRectangle(const string name,
                         const datetime time1, const double price1,
                         const datetime time2, const double price2,
                         const color render_color)
{
    if(!ObjectCreate(0, name, OBJ_RECTANGLE, 0, time1, price1, time2, price2))
    {
        Print("ZoneOverlay: ObjectCreate failed for '", name,
              "', err=", GetLastError());
        return;
    }
    ObjectSetInteger(0, name, OBJPROP_COLOR, render_color);
    ObjectSetInteger(0, name, OBJPROP_BGCOLOR, render_color);
    ObjectSetInteger(0, name, OBJPROP_FILL, true);
    ObjectSetInteger(0, name, OBJPROP_BACK, true);          // behind candles
    ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);   // can't drag
    ObjectSetInteger(0, name, OBJPROP_SELECTED, false);
    ObjectSetInteger(0, name, OBJPROP_HIDDEN, false);       // shows in list
    ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
    ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
}


//+------------------------------------------------------------------+
//| Refresh the right edge + color of an existing rectangle.         |
//| (Zone bounds rarely change; this mostly extends the right edge    |
//| forward and reflects status changes like CONFIRMED → ACTIVE.)    |
//+------------------------------------------------------------------+
void UpdateZoneRectangle(const string name,
                         const datetime new_time2,
                         const double new_price2,
                         const color new_color)
{
    // ObjectMove uses corner indices: 0 = (time1, price1), 1 = (time2, price2)
    ObjectMove(0, name, 1, new_time2, new_price2);
    // Update colors so CONFIRMED → ACTIVE → FLIPPED transitions show.
    ObjectSetInteger(0, name, OBJPROP_COLOR, new_color);
    ObjectSetInteger(0, name, OBJPROP_BGCOLOR, new_color);
}


//+------------------------------------------------------------------+
//| Pick the color for a zone based on its direction + status.       |
//+------------------------------------------------------------------+
color PickColor(const string direction, const string status,
                const string flipped_direction)
{
    if(status == "FLIPPED")
        return InpColorFlipped;
    if(status == "ACTIVE")
        return (direction == "BUY") ? InpColorActiveBuy : InpColorActiveSell;
    // CONFIRMED (default).
    return (direction == "BUY") ? InpColorBuy : InpColorSell;
}


//+------------------------------------------------------------------+
//| Delete every BOTZONE_* object NOT in the keep_names list.        |
//+------------------------------------------------------------------+
void DeleteOrphanZones(const string &keep_names[])
{
    int total = ObjectsTotal(0, 0, OBJ_RECTANGLE);
    for(int i = total - 1; i >= 0; i--)
    {
        string obj_name = ObjectName(0, i, 0, OBJ_RECTANGLE);
        if(StringFind(obj_name, OBJ_PREFIX) != 0)
            continue;
        bool keep = false;
        for(int k = 0; k < ArraySize(keep_names); k++)
        {
            if(keep_names[k] == obj_name)
            {
                keep = true;
                break;
            }
        }
        if(!keep)
        {
            ObjectDelete(0, obj_name);
        }
    }
}


//+------------------------------------------------------------------+
//| Delete every BOTZONE_* (used on EA shutdown when cleanup is on). |
//+------------------------------------------------------------------+
void DeleteAllBotZones()
{
    int total = ObjectsTotal(0, 0, OBJ_RECTANGLE);
    for(int i = total - 1; i >= 0; i--)
    {
        string obj_name = ObjectName(0, i, 0, OBJ_RECTANGLE);
        if(StringFind(obj_name, OBJ_PREFIX) == 0)
            ObjectDelete(0, obj_name);
    }
}
