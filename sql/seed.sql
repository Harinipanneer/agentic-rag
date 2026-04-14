-- =============================================================================
-- sql/seed.sql — Bootstrap agentic_rag_db with sample e-commerce schema
--
-- Run as a PostgreSQL superuser:
--   psql -U postgres -f sql/seed.sql
--
-- What this script does:
--   1. Creates the agentic_rag_db database.
--   2. Creates a read-only role (rag_readonly) for the NL2SQL agent.
--   3. Defines the schema: categories, products, orders, order_items.
--   4. Seeds realistic sample data.
--   5. Grants SELECT-only privileges to rag_readonly.
--   6. Enforces read-only at the session level for rag_readonly.
-- =============================================================================

-- ── Step 1: Create DB (run connected to the default 'postgres' database) ─────

-- Drop and recreate for a clean slate (comment out in production)
DROP DATABASE IF EXISTS agentic_rag_db;
CREATE DATABASE agentic_rag_db;

-- ── Step 2: Create read-only user ────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rag_readonly') THEN
        CREATE USER rag_readonly WITH LOGIN ENCRYPTED PASSWORD 'rag_readonly_pass';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE agentic_rag_db TO rag_readonly;

-- ── Switch to agentic_rag_db ──────────────────────────────────────────────────

\c agentic_rag_db

-- ── Step 3: Schema ────────────────────────────────────────────────────────────

GRANT USAGE ON SCHEMA public TO rag_readonly;

-- -----------------------------------------------------------------------------
-- categories
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS categories (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- products
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id             SERIAL PRIMARY KEY,
    name           VARCHAR(200)   NOT NULL,
    category_id    INT            REFERENCES categories(id),
    price          NUMERIC(10, 2) NOT NULL,
    stock_quantity INT            DEFAULT 0,
    description    TEXT,
    is_active      BOOLEAN        DEFAULT TRUE,
    created_at     TIMESTAMPTZ    DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- orders
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id             SERIAL PRIMARY KEY,
    customer_name  VARCHAR(200)   NOT NULL,
    customer_email VARCHAR(200),
    total_amount   NUMERIC(10, 2),
    status         VARCHAR(50)    DEFAULT 'pending',
    -- status values: pending, confirmed, shipped, delivered, cancelled
    created_at     TIMESTAMPTZ    DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- order_items
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INT            REFERENCES orders(id),
    product_id INT            REFERENCES products(id),
    quantity   INT            NOT NULL,
    unit_price NUMERIC(10, 2) NOT NULL
);

-- ── Step 4: Seed data ─────────────────────────────────────────────────────────

-- ── Categories (7) ────────────────────────────────────────────────────────────
INSERT INTO categories (name, description) VALUES
    ('Electronics',     'Gadgets, consumer electronics, and accessories'),
    ('Books',           'Fiction, non-fiction, technical, and educational books'),
    ('Clothing',        'Men''s, women''s, and children''s apparel'),
    ('Home & Kitchen',  'Furniture, cookware, and home décor'),
    ('Sports & Outdoors', 'Fitness equipment, outdoor gear, and sportswear'),
    ('Toys & Games',    'Board games, puzzles, and children''s toys'),
    ('Health & Beauty', 'Personal care, vitamins, and wellness products');

-- ── Products (25) ─────────────────────────────────────────────────────────────
INSERT INTO products (name, category_id, price, stock_quantity, description, is_active) VALUES
    -- Electronics (category 1)
    ('Wireless Noise-Cancelling Headphones',  1, 149.99,  85,  'Over-ear headphones with 30-hour battery and active noise cancellation', TRUE),
    ('Mechanical Keyboard — TKL',             1,  89.99, 120,  'Tenkeyless mechanical keyboard with blue switches and RGB backlighting', TRUE),
    ('4K USB-C Monitor 27"',                  1, 329.99,  40,  '27-inch 4K IPS display with USB-C 90 W power delivery', TRUE),
    ('Smart Home Hub',                        1,  59.99,  60,  'Zigbee + Z-Wave compatible hub, works with Alexa and Google Home', TRUE),
    ('Portable SSD 1 TB',                     1,  99.99, 200,  'NVMe portable SSD with read speeds up to 1050 MB/s', TRUE),

    -- Books (category 2)
    ('Clean Code by Robert C. Martin',        2,  34.99, 300,  'A handbook of agile software craftsmanship', TRUE),
    ('Designing Data-Intensive Applications', 2,  44.99, 250,  'Principles, practices, and patterns for scalable systems', TRUE),
    ('The Pragmatic Programmer',              2,  39.99, 280,  '20th anniversary edition — your journey to mastery', TRUE),

    -- Clothing (category 3)
    ('Men''s Slim-Fit Chinos',                3,  49.99, 500,  'Stretch chinos available in 12 colours, sizes 28–40', TRUE),
    ('Women''s Running Jacket',               3,  79.99, 160,  'Lightweight wind-proof running jacket with reflective trim', TRUE),
    ('Unisex Organic Cotton T-Shirt',         3,  24.99, 800,  '100 % GOTS-certified organic cotton, available in 8 colours', TRUE),

    -- Home & Kitchen (category 4)
    ('Stainless Steel Cookware Set 10-piece',  4, 189.99,  70,  'Tri-ply stainless steel, oven-safe to 260 °C', TRUE),
    ('Bamboo Cutting Board Set',              4,  34.99, 150,  'Set of 3 boards with juice grooves and non-slip feet', TRUE),
    ('Air Purifier HEPA H13',                 4, 129.99,  55,  'Covers up to 40 m², removes 99.97 % of airborne particles', TRUE),

    -- Sports & Outdoors (category 5)
    ('Adjustable Dumbbell Set 5–52.5 lb',     5, 299.99,  30,  '15 weight settings replace 15 pairs of dumbbells', TRUE),
    ('Hiking Backpack 50 L',                  5,  89.99,  90,  'Waterproof, with integrated rain cover and hip-belt pockets', TRUE),
    ('Yoga Mat Premium 6 mm',                 5,  39.99, 220,  'Non-slip, eco-friendly TPE foam yoga mat', TRUE),

    -- Toys & Games (category 6)
    ('Classic Chess Set Wooden',              6,  59.99, 100,  'Hand-carved pieces with roll-up board, box included', TRUE),
    ('LEGO Technic Bugatti Chiron',           6, 369.99,  20,  '3 599-piece model, 1:8 scale, with working W16 engine', TRUE),
    ('Card Game — Exploding Kittens',         6,  19.99, 350,  'A highly strategic kitty-powered card game', TRUE),

    -- Health & Beauty (category 7)
    ('Vitamin D3 + K2 Supplement 90 caps',   7,  18.99, 400,  '2000 IU D3 + 100 mcg K2 MK-7 per capsule', TRUE),
    ('Electric Toothbrush Sonic',             7,  49.99, 130,  '62 000 strokes/min, 3 modes, 30-day battery', TRUE),
    ('Natural Face Serum Vitamin C 30 ml',    7,  28.99, 180,  '20 % L-ascorbic acid + hyaluronic acid', TRUE),
    ('Resistance Band Set 5-loop',            5,  21.99, 310,  'Five resistance levels (5–50 lb), latex-free', TRUE),
    ('Smart Water Bottle 500 ml',             1,  34.99,  75,  'Tracks hydration via Bluetooth app, keeps cold 24 h / hot 12 h', FALSE);

-- ── Orders (12) ───────────────────────────────────────────────────────────────
INSERT INTO orders (customer_name, customer_email, total_amount, status, created_at) VALUES
    ('Alice Johnson',   'alice@example.com',   389.97, 'delivered',  NOW() - INTERVAL '30 days'),
    ('Bob Smith',       'bob@example.com',     239.98, 'delivered',  NOW() - INTERVAL '25 days'),
    ('Carol Williams',  'carol@example.com',    74.98, 'shipped',    NOW() - INTERVAL '10 days'),
    ('David Brown',     'david@example.com',   639.98, 'confirmed',  NOW() - INTERVAL '5 days'),
    ('Eva Martinez',    'eva@example.com',     104.97, 'pending',    NOW() - INTERVAL '2 days'),
    ('Frank Lee',       'frank@example.com',   479.97, 'delivered',  NOW() - INTERVAL '45 days'),
    ('Grace Kim',       'grace@example.com',   189.99, 'cancelled',  NOW() - INTERVAL '15 days'),
    ('Henry Davis',     'henry@example.com',   159.98, 'shipped',    NOW() - INTERVAL '7 days'),
    ('Isabella Garcia', 'isabella@example.com', 54.98, 'pending',    NOW() - INTERVAL '1 day'),
    ('Jack Wilson',     'jack@example.com',    599.97, 'confirmed',  NOW() - INTERVAL '3 days'),
    ('Karen Thomas',    'karen@example.com',    62.97, 'delivered',  NOW() - INTERVAL '60 days'),
    ('Liam Anderson',   'liam@example.com',    369.99, 'shipped',    NOW() - INTERVAL '6 days');

-- ── Order items ───────────────────────────────────────────────────────────────
-- Using sub-selects to avoid hard-coded IDs.

DO $$
DECLARE
    -- Product IDs looked up by name
    p_headphones     INT := (SELECT id FROM products WHERE name = 'Wireless Noise-Cancelling Headphones');
    p_kb             INT := (SELECT id FROM products WHERE name = 'Mechanical Keyboard — TKL');
    p_monitor        INT := (SELECT id FROM products WHERE name = '4K USB-C Monitor 27"');
    p_ssd            INT := (SELECT id FROM products WHERE name = 'Portable SSD 1 TB');
    p_clean_code     INT := (SELECT id FROM products WHERE name = 'Clean Code by Robert C. Martin');
    p_ddia           INT := (SELECT id FROM products WHERE name = 'Designing Data-Intensive Applications');
    p_pragmatic      INT := (SELECT id FROM products WHERE name = 'The Pragmatic Programmer');
    p_chinos         INT := (SELECT id FROM products WHERE name = 'Men''s Slim-Fit Chinos');
    p_jacket         INT := (SELECT id FROM products WHERE name = 'Women''s Running Jacket');
    p_tshirt         INT := (SELECT id FROM products WHERE name = 'Unisex Organic Cotton T-Shirt');
    p_cookware       INT := (SELECT id FROM products WHERE name = 'Stainless Steel Cookware Set 10-piece');
    p_dumbbells      INT := (SELECT id FROM products WHERE name = 'Adjustable Dumbbell Set 5–52.5 lb');
    p_backpack       INT := (SELECT id FROM products WHERE name = 'Hiking Backpack 50 L');
    p_yoga           INT := (SELECT id FROM products WHERE name = 'Yoga Mat Premium 6 mm');
    p_chess          INT := (SELECT id FROM products WHERE name = 'Classic Chess Set Wooden');
    p_lego           INT := (SELECT id FROM products WHERE name = 'LEGO Technic Bugatti Chiron');
    p_cards          INT := (SELECT id FROM products WHERE name = 'Card Game — Exploding Kittens');
    p_vitamin        INT := (SELECT id FROM products WHERE name = 'Vitamin D3 + K2 Supplement 90 caps');
    p_toothbrush     INT := (SELECT id FROM products WHERE name = 'Electric Toothbrush Sonic');
    p_serum          INT := (SELECT id FROM products WHERE name = 'Natural Face Serum Vitamin C 30 ml');
    p_resistance     INT := (SELECT id FROM products WHERE name = 'Resistance Band Set 5-loop');
    p_smart_hub      INT := (SELECT id FROM products WHERE name = 'Smart Home Hub');

    -- Order IDs (inserted in sequence — rely on serial)
    o1 INT := (SELECT id FROM orders WHERE customer_email = 'alice@example.com');
    o2 INT := (SELECT id FROM orders WHERE customer_email = 'bob@example.com');
    o3 INT := (SELECT id FROM orders WHERE customer_email = 'carol@example.com');
    o4 INT := (SELECT id FROM orders WHERE customer_email = 'david@example.com');
    o5 INT := (SELECT id FROM orders WHERE customer_email = 'eva@example.com');
    o6 INT := (SELECT id FROM orders WHERE customer_email = 'frank@example.com');
    o7 INT := (SELECT id FROM orders WHERE customer_email = 'grace@example.com');
    o8 INT := (SELECT id FROM orders WHERE customer_email = 'henry@example.com');
    o9 INT := (SELECT id FROM orders WHERE customer_email = 'isabella@example.com');
    o10 INT := (SELECT id FROM orders WHERE customer_email = 'jack@example.com');
    o11 INT := (SELECT id FROM orders WHERE customer_email = 'karen@example.com');
    o12 INT := (SELECT id FROM orders WHERE customer_email = 'liam@example.com');
BEGIN
    INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
        -- Alice: headphones + monitor + clean_code
        (o1, p_headphones, 1, 149.99),
        (o1, p_monitor,    1, 329.99),  -- total would be slightly off; orders.total_amount is pre-set
        (o1, p_clean_code, 1,  34.99),

        -- Bob: keyboard + ssd
        (o2, p_kb,  1,  89.99),
        (o2, p_ssd, 1,  99.99),
        (o2, p_ddia, 1,  44.99),

        -- Carol: jacket + tshirt
        (o3, p_jacket, 1, 79.99),
        (o3, p_tshirt, 1, 24.99),

        -- David: monitor + dumbbells (big order)
        (o4, p_monitor,   1, 329.99),
        (o4, p_dumbbells, 1, 299.99),

        -- Eva: clean_code + pragmatic + vitamin
        (o5, p_clean_code, 1, 34.99),
        (o5, p_pragmatic,  1, 39.99),
        (o5, p_vitamin,    1, 18.99),

        -- Frank: headphones + backpack + chess
        (o6, p_headphones, 1, 149.99),
        (o6, p_backpack,   1,  89.99),
        (o6, p_chess,      1,  59.99),
        (o6, p_yoga,       2,  39.99),

        -- Grace: cookware
        (o7, p_cookware, 1, 189.99),

        -- Henry: chinos + jacket
        (o8, p_chinos, 1,  49.99),
        (o8, p_jacket, 1,  79.99),
        (o8, p_cards,  1,  19.99),

        -- Isabella: vitamin + serum
        (o9, p_vitamin, 1, 18.99),
        (o9, p_serum,   1, 28.99),

        -- Jack: ssd + lego + smart_hub
        (o10, p_ssd,       1,  99.99),
        (o10, p_lego,      1, 369.99),
        (o10, p_smart_hub, 1,  59.99),
        (o10, p_toothbrush,1,  49.99),

        -- Karen: cards + tshirt + resistance
        (o11, p_cards,      1, 19.99),
        (o11, p_tshirt,     1, 24.99),
        (o11, p_resistance, 1, 21.99),

        -- Liam: lego
        (o12, p_lego, 1, 369.99);
END;
$$;

-- ── Step 5: Grant read-only privileges ────────────────────────────────────────

GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO rag_readonly;

-- ── Step 6: Enforce read-only at session level for safety ────────────────────

ALTER USER rag_readonly SET default_transaction_read_only = on;

-- ── Verification ──────────────────────────────────────────────────────────────

SELECT 'categories'  AS tbl, COUNT(*) AS rows FROM categories
UNION ALL
SELECT 'products',   COUNT(*) FROM products
UNION ALL
SELECT 'orders',     COUNT(*) FROM orders
UNION ALL
SELECT 'order_items',COUNT(*) FROM order_items;
