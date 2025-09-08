# Training Plan: Databricks AI/BI Genie Workshop with Sales \& Customer Data

## 0. Setup: Create Databricks Free Edition Account

### Create Your Free Account

**Step 1:** Visit the Databricks Free Edition signup page

**Step 2:** Choose your preferred signup method (Google, Microsoft, or email)

**Step 3:** Complete account verification - Databricks will immediately create your workspace

**Step 4:** Access your workspace through the provided URL

### Initial Workspace Setup

**Step 5:** Navigate to the workspace interface and familiarize yourself with the sidebar navigation

**Step 6:** Locate the "Data" section in the sidebar to prepare for data ingestion

## 1. Data Ingestion: Create Catalog, Schema \& Upload Related Tables

### Create Unity Catalog Structure via UI

**Step 1:** In the Databricks workspace sidebar, click "Catalog"

**Step 2:** Click "Catalogs" below Quick access

**Step 3:** Click "Create catalog"

**Step 4:** In the "Create a new catalog" dialog:

- Enter Catalog name: `sales_analytics`
- Select Type: Standard (default)
- Add Comment: Catalog for sales and customer analytics (optional)

**Step 5:** Click "Create" to create your catalog

### Create Schema via UI

**Step 1:** Click on your newly created sales_analytics catalog

**Step 2:** Click "Create Schema"

**Step 3:** In the schema creation dialog:

- Schema name: `sales_data`
- Comment: Schema for raw sales and customer data (optional)

**Step 4:** Click "Create" to create your schema

### Upload Sales Transactions Table

**Step 1:** Create `sales_transactions.csv` on your local machine:

```csv
order_id,customer_id,product_name,category,quantity,unit_price,total_amount,order_date,sales_rep,region
1001,CUST_001,Wireless Headphones,Electronics,2,99.99,199.98,2024-01-15,John Smith,North
1002,CUST_002,Office Chair,Furniture,1,299.99,299.99,2024-01-16,Jane Doe,South
1003,CUST_003,Coffee Machine,Appliances,1,149.99,149.99,2024-01-17,John Smith,East
1004,CUST_001,Gaming Mouse,Electronics,2,79.99,159.98,2024-01-18,Sarah Wilson,North
1005,CUST_004,Standing Desk,Furniture,1,599.99,599.99,2024-01-19,Mike Johnson,West
1006,CUST_003,Laptop Stand,Electronics,1,45.00,45.00,2024-01-20,Jane Doe,East
1007,CUST_002,Coffee Mug,Appliances,4,9.99,39.96,2024-01-21,John Smith,South
1008,CUST_004,Bluetooth Speaker,Electronics,1,129.99,129.99,2024-01-22,Sarah Wilson,West
1009,CUST_001,Ergonomic Keyboard,Electronics,1,89.99,89.99,2024-01-23,Mike Johnson,North
1010,CUST_002,Table Lamp,Furniture,2,34.99,69.98,2024-01-24,Jane Doe,South
```

**Step 2:** In the schema, click "create" → "table"

**Step 3:** Click "browse", upload the file

**Step 4:** Configure:

- Catalog: `sales_analytics`
- Schema: `sales_data`
- Table name: `sales_transactions`

**Step 5:** Click "Create" to create the table

### Upload Customers Table

**Step 6:** Create `customers.csv` on your local machine:

```csv
customer_id,customer_name,customer_email,city,signup_date,customer_segment
CUST_001,Alice Smith,alice.smith@example.com,Springfield,2023-08-15,Premium
CUST_002,Bob Jones,bob.jones@example.com,Rivertown,2023-09-20,Standard
CUST_003,Carol Lee,carol.lee@example.com,Greenville,2023-10-05,Premium
CUST_004,David Kim,david.kim@example.com,Lakeside,2023-11-12,Standard
```

**Step 7:** Repeat the upload process for customers table:

- Table name: `customers`
- Same catalog and schema: `sales_analytics.sales_data`

**Step 8:** Verify both tables exist by navigating to Catalog → sales_analytics → sales_data

## 2. Data Exploration with Genie: Single Table \& Join Queries

### Basic Single-Table Analysis

**Step 1:** Open Genie from the AI/BI section in the sidebar
**Step 2:** Connect to your sales_analytics.sales_data schema and select all tables

### Basic Sales Analysis Prompts:

- "What is the total revenue from all sales?"
- "Show me the top 5 best-selling products by quantity"
- "What is the average order value per transaction?"
- "Display sales performance by region as a bar chart"


### Customer Table Analysis:

- "How many customers do we have?"
- "Show customer distribution by city"
- "What is the breakdown of customers by segment?"


### Advanced Join-Based Analysis with Genie

**Step 3:** Use Genie's natural language capability for complex joins:

### Customer Sales Analysis:

- "Show total sales by customer name"
- "Which customers have spent the most money?"
- "Create a bar chart of revenue by customer city"
- "Display customer lifetime value with their email addresses"


### Geographic Sales Insights:

- "Compare sales performance across different customer cities"
- "Show average order value by customer city"
- "Which cities generate the most revenue?"


### Customer Behavior Analysis:

- "Show customers who have made multiple purchases"
- "Display purchase frequency by customer segment"
- "Which Premium customers have the highest sales?"


### Iterative Query Refinement

**Step 4:** Practice refining complex queries:

- "Filter the customer sales analysis for Electronics category only"
- "Show only Premium customers from the previous analysis"
- "Compare Standard vs Premium customer purchase patterns"


## 3A. Advanced Data Setup with SQL (30 minutes)

### Create Additional Tables

**Step 1:** Go to query

**Step 2:** Create Query

**Step 3:** Create Products Table with Cost Information

```sql
CREATE TABLE sales_analytics.sales_data.products (
  product_id STRING,
  product_name STRING,
  category STRING,
  unit_price DECIMAL(10,2),
  cost DECIMAL(10,2),
  brand STRING,
  launch_date DATE
) USING DELTA;

INSERT INTO sales_analytics.sales_data.products VALUES
  ('PROD_001', 'Wireless Headphones', 'Electronics', 99.99, 65.00, 'Brand A', '2023-06-01'),
  ('PROD_002', 'Office Chair', 'Furniture', 299.99, 180.00, 'Brand B', '2023-07-15'),
  ('PROD_003', 'Coffee Machine', 'Appliances', 149.99, 95.00, 'Brand C', '2023-08-10'),
  ('PROD_004', 'Gaming Mouse', 'Electronics', 79.99, 50.00, 'Brand A', '2023-09-05'),
  ('PROD_005', 'Standing Desk', 'Furniture', 599.99, 350.00, 'Brand B', '2023-10-20'),
  ('PROD_006', 'Laptop Stand', 'Electronics', 45.00, 25.00, 'Brand D', '2023-11-01'),
  ('PROD_007', 'Coffee Mug', 'Appliances', 9.99, 4.00, 'Brand C', '2023-12-01'),
  ('PROD_008', 'Bluetooth Speaker', 'Electronics', 129.99, 75.00, 'Brand A', '2024-01-01'),
  ('PROD_009', 'Ergonomic Keyboard', 'Electronics', 89.99, 55.00, 'Brand A', '2024-01-10'),
  ('PROD_010', 'Table Lamp', 'Furniture', 34.99, 20.00, 'Brand E', '2024-01-15'),
  ('PROD_011', 'Wireless Mouse', 'Electronics', 29.99, 18.00, 'Brand D', '2024-01-20'),
  ('PROD_012', 'Monitor Stand', 'Electronics', 79.99, 45.00, 'Brand B', '2024-01-25'),
  ('PROD_013', 'Desk Organizer', 'Furniture', 24.99, 12.00, 'Brand E', '2024-02-01'),
  ('PROD_014', 'Coffee Grinder', 'Appliances', 89.99, 60.00, 'Brand C', '2024-02-05'),
  ('PROD_015', 'USB Hub', 'Electronics', 39.99, 22.00, 'Brand D', '2024-02-10');
```

**Step 4:** Create Sales Targets Table

```sql
CREATE TABLE sales_analytics.sales_data.sales_targets (
  sales_rep STRING,
  region STRING,
  target_amount DECIMAL(10,2),
  target_period STRING,
  year INT
) USING DELTA;

INSERT INTO sales_analytics.sales_data.sales_targets VALUES
  ('John Smith', 'North', 2000.00, '2024-Q1', 2024),
  ('Jane Doe', 'South', 1800.00, '2024-Q1', 2024),
  ('Mike Johnson', 'West', 2100.00, '2024-Q1', 2024),
  ('Sarah Wilson', 'East', 1750.00, '2024-Q1', 2024),
  ('John Smith', 'East', 1600.00, '2024-Q1', 2024),
  ('Jane Doe', 'East', 1400.00, '2024-Q1', 2024),
  ('Sarah Wilson', 'West', 1900.00, '2024-Q1', 2024),
  ('Sarah Wilson', 'North', 1700.00, '2024-Q1', 2024);
```


## 3B. Create Strategic Analytics Views

**Step 5:** Build Comprehensive Sales Analytics View

```sql
CREATE SCHEMA IF NOT EXISTS sales_analytics.views;

CREATE OR REPLACE VIEW sales_analytics.views.comprehensive_sales AS
SELECT 
    s.order_id,
    s.order_date,
    s.quantity,
    s.total_amount,
    -- Customer information
    c.customer_name,
    c.customer_email,
    c.city,
    -- Product information
    p.product_id,
    p.product_name,
    p.category,
    p.brand,
    p.unit_price,
    p.cost,
    p.launch_date,
    -- Sales rep and targets
    s.sales_rep,
    s.region,
    t.target_amount as quarterly_target,
    -- Calculated financial metrics
    (p.unit_price - p.cost) * s.quantity as gross_profit,
    ((p.unit_price - p.cost) / p.unit_price) * 100 as profit_margin_pct,
    s.total_amount - (p.cost * s.quantity) as profit_amount,
    -- Time dimensions for analysis
    EXTRACT(MONTH FROM s.order_date) as order_month,
    EXTRACT(YEAR FROM s.order_date) as order_year,
    EXTRACT(QUARTER FROM s.order_date) as order_quarter,
    DAYNAME(s.order_date) as day_name,
    WEEKOFYEAR(s.order_date) as week_number
FROM sales_analytics.sales_data.sales_transactions s
LEFT JOIN sales_analytics.sales_data.customers c ON s.customer_id = c.customer_id
LEFT JOIN sales_analytics.sales_data.products p ON s.product_name = p.product_name
LEFT JOIN sales_analytics.sales_data.sales_targets t ON s.sales_rep = t.sales_rep 
    AND s.region = t.region AND t.year = 2024
WHERE s.total_amount > 0;
```

**Step 6:** Create Customer Intelligence View

```sql
CREATE OR REPLACE VIEW sales_analytics.views.customer_intelligence AS
WITH customer_metrics AS (
  SELECT 
    c.customer_id,
    c.customer_name,
    c.city,
    -- Purchase behavior
    COUNT(s.order_id) as total_orders,
    SUM(s.total_amount) as lifetime_value,
    AVG(s.total_amount) as avg_order_value,
    SUM(s.quantity) as total_items_purchased,
    -- Profitability analysis
    SUM((p.unit_price - p.cost) * s.quantity) as customer_gross_profit,
    ROUND(AVG(((p.unit_price - p.cost) / p.unit_price) * 100), 2) as avg_profit_margin,
    -- Timing analysis
    MIN(s.order_date) as first_purchase_date,
    MAX(s.order_date) as last_purchase_date,
    DATEDIFF(CURRENT_DATE(), MAX(s.order_date)) as days_since_last_order,
    -- Product preferences
    COUNT(DISTINCT p.category) as categories_purchased,
    COUNT(DISTINCT p.brand) as brands_purchased
  FROM sales_analytics.sales_data.customers c
  LEFT JOIN sales_analytics.sales_data.sales_transactions s ON c.customer_id = s.customer_id
  LEFT JOIN sales_analytics.sales_data.products p ON s.product_name = p.product_name
  GROUP BY c.customer_id, c.customer_name, c.city
)
SELECT 
  *,
  -- Advanced customer segmentation
  CASE 
    WHEN lifetime_value >= 1000 AND total_orders >= 4 THEN 'VIP Customer'
    WHEN lifetime_value >= 500 AND total_orders >= 3 THEN 'High Value Customer'
    WHEN lifetime_value >= 200 OR total_orders >= 2 THEN 'Regular Customer'
    ELSE 'New Customer'
  END as customer_segment,
  -- Churn risk assessment
  CASE 
    WHEN days_since_last_order > 60 THEN 'High Risk'
    WHEN days_since_last_order > 30 THEN 'Medium Risk'
    WHEN days_since_last_order > 15 THEN 'Low Risk'
    ELSE 'Active'
  END as churn_risk,
  -- Customer health score (0-100)
  ROUND(
    (CASE WHEN days_since_last_order <= 15 THEN 40 
          WHEN days_since_last_order <= 30 THEN 30 
          WHEN days_since_last_order <= 60 THEN 20 
          ELSE 10 END) + 
    (CASE WHEN total_orders >= 5 THEN 30 
          WHEN total_orders >= 3 THEN 25 
          WHEN total_orders >= 2 THEN 20 
          ELSE 15 END) + 
    (CASE WHEN lifetime_value >= 1000 THEN 30 
          WHEN lifetime_value >= 500 THEN 25 
          WHEN lifetime_value >= 200 THEN 20 
          ELSE 15 END), 0
  ) as customer_health_score
FROM customer_metrics;
```

**Step 7:** Create Sales Performance Analytics View

```sql
CREATE OR REPLACE VIEW sales_analytics.views.sales_performance AS
SELECT 
    s.sales_rep,
    s.region,
    MAX(t.target_amount) as quarterly_target,
    -- Performance metrics
    COUNT(s.order_id) as total_deals,
    SUM(s.total_amount) as actual_revenue,
    AVG(s.total_amount) as avg_deal_size,
    COUNT(DISTINCT s.customer_id) as unique_customers,
    -- Target achievement analysis
    ROUND((SUM(s.total_amount) / MAX(t.target_amount)) * 100, 2) as target_achievement_pct,
    SUM(s.total_amount) - MAX(t.target_amount) as variance_from_target,
    CASE 
        WHEN (SUM(s.total_amount) / MAX(t.target_amount)) >= 1.1 THEN 'Exceeds Target'
        WHEN (SUM(s.total_amount) / MAX(t.target_amount)) >= 0.9 THEN 'Meets Target'
        ELSE 'Below Target'
    END as performance_status,
    -- Product mix analysis
    COUNT(DISTINCT p.category) as categories_sold,
    COUNT(DISTINCT p.brand) as brands_sold,
    -- Profitability metrics
    SUM((p.unit_price - p.cost) * s.quantity) as total_gross_profit,
    ROUND(AVG(((p.unit_price - p.cost) / p.unit_price) * 100), 2) as avg_profit_margin,
    -- Efficiency metrics
    ROUND(SUM(s.total_amount) / COUNT(s.order_id), 2) as revenue_per_deal,
    ROUND(SUM(s.total_amount) / COUNT(DISTINCT s.customer_id), 2) as revenue_per_customer
FROM sales_analytics.sales_data.sales_transactions s
LEFT JOIN sales_analytics.sales_data.sales_targets t ON s.sales_rep = t.sales_rep 
    AND s.region = t.region AND t.year = 2024
LEFT JOIN sales_analytics.sales_data.products p ON s.product_name = p.product_name
GROUP BY s.sales_rep, s.region
HAVING MAX(t.target_amount) IS NOT NULL;
```

**Step 8:** Create Product Analytics View

```sql
CREATE OR REPLACE VIEW sales_analytics.views.product_analytics AS
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.brand,
    p.unit_price,
    p.cost,
    p.launch_date,
    -- Sales performance
    COALESCE(COUNT(s.order_id), 0) as times_sold,
    COALESCE(SUM(s.quantity), 0) as units_sold,
    COALESCE(SUM(s.total_amount), 0) as product_revenue,
    -- Profitability analysis
    p.unit_price - p.cost as unit_profit,
    ROUND(((p.unit_price - p.cost) / p.unit_price) * 100, 2) as profit_margin_pct,
    COALESCE(SUM((p.unit_price - p.cost) * s.quantity), 0) as total_profit,
    -- Market performance
    RANK() OVER (PARTITION BY p.category ORDER BY COALESCE(SUM(s.total_amount), 0) DESC) as category_rank,
    RANK() OVER (ORDER BY COALESCE(SUM(s.total_amount), 0) DESC) as overall_rank,
    -- Geographic distribution
    COUNT(DISTINCT c.city) as cities_sold_in,
    COUNT(DISTINCT s.region) as regions_sold_in
FROM sales_analytics.sales_data.products p
LEFT JOIN sales_analytics.sales_data.sales_transactions s ON p.product_name = s.product_name
LEFT JOIN sales_analytics.sales_data.customers c ON s.customer_id = c.customer_id
GROUP BY p.product_id, p.product_name, p.category, p.brand, p.unit_price, p.cost, p.launch_date;
```


## 3C. Multi-Page Dashboard Creation

### Page 1: Executive Dashboard

**Dataset:** Use comprehensive_sales view

### KPI Cards Row:

```sql
-- Total Revenue KPI
SELECT ROUND(SUM(total_amount), 2) as total_revenue FROM comprehensive_sales;

-- Gross Profit KPI  
SELECT ROUND(SUM(gross_profit), 2) as total_gross_profit FROM comprehensive_sales;

-- Average Profit Margin KPI
SELECT ROUND(AVG(profit_margin_pct), 1) as avg_profit_margin FROM comprehensive_sales;

-- Active Customers KPI
SELECT COUNT(DISTINCT customer_name) as active_customers FROM comprehensive_sales;
```


### Main Visualizations:

**Revenue Trend by Month:**

- Chart Type: Line Chart
- X-axis: order_date (monthly aggregation)
- Y-axis: SUM(total_amount)
- Secondary Y-axis: SUM(gross_profit)

**Top Performing Products:**

- Chart Type: Horizontal Bar Chart
- X-axis: SUM(total_amount)
- Y-axis: product_name
- Color: category
- Limit: Top 10

**Regional Performance Map:**

- Chart Type: Bar Chart
- X-axis: region
- Y-axis: SUM(total_amount)
- Color: AVG(profit_margin_pct)


### Page 2: Customer Analytics

**Dataset:** Use customer_intelligence view

**Customer Segmentation:**

```sql
-- Customer Distribution by Segment
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(lifetime_value), 2) as avg_ltv
FROM customer_intelligence 
GROUP BY customer_segment;
```


### Visualizations:

- Pie Chart: Customer distribution by segment
- Scatter Plot: Lifetime value vs. Total orders (colored by segment)
- Bar Chart: Top 10 customers by lifetime value
- Heatmap: Churn risk by city


### Page 3: Sales Performance

**Dataset:** Use sales_performance view

**Sales Rep Performance Matrix:**

```sql
-- Sales Rep Achievement Analysis
SELECT 
    sales_rep,
    region,
    actual_revenue,
    quarterly_target,
    target_achievement_pct,
    total_gross_profit,
    unique_customers
FROM sales_performance 
ORDER BY target_achievement_pct DESC;
```


### Visualizations:

- Gauge Charts: Target achievement % for each sales rep
- Waterfall Chart: Revenue contribution by sales rep
- Combo Chart: Revenue vs. Profit by region
- Table: Detailed performance metrics


### Page 4: Product \& Brand Analysis

**Dataset:** Use product_analytics view

**Brand Performance Analysis:**

```sql
-- Brand Profitability Analysis
SELECT 
    brand,
    category,
    SUM(units_sold) as total_units_sold,
    SUM(product_revenue) as total_revenue,
    SUM(total_profit) as total_profit,
    ROUND(AVG(profit_margin_pct), 2) as avg_margin
FROM product_analytics
GROUP BY brand, category
ORDER BY total_profit DESC;
```


### Visualizations:

- Stacked Bar Chart: Revenue by brand, stacked by category
- Bubble Chart: Profit margin vs. Revenue (size = units sold)
- Heatmap: Brand performance by region
- Tree Map: Market share by brand and category


## Advanced Interactive Features Implementation

### Global Filters (Apply to All Pages):

1. Date Range Filter: Last 30, 60, 90 days, or custom range
2. Region Filter: Multi-select for North, South, East, West
3. Category Filter: Electronics, Furniture, Appliances
4. Brand Filter: Brand A, Brand B, Brand C, Brand D, Brand E

### Cross-Filtering Examples:

- Click on a region → All charts filter to that region's data
- Select customer segment → Product analysis updates to show preferences
- Choose brand → Customer and sales rep data filters accordingly
- Pick time period → All KPIs and trends adjust automatically


## Enhanced Step 9: AI-Powered Analysis with ai_query Function (20 minutes)

Create Customer Reviews data : 

```sql
-- Add customer feedback table
CREATE TABLE sales_analytics.sales_data.customer_reviews (
  review_id STRING,
  customer_id STRING,
  product_name STRING,
  review_text STRING,
  review_date DATE
) USING DELTA;

INSERT INTO sales_analytics.sales_data.customer_reviews VALUES
  ('REV_001', 'CUST_001', 'Wireless Headphones', 'Amazing sound quality and comfortable fit. Love these headphones!', '2024-01-20'),
  ('REV_002', 'CUST_002', 'Office Chair', 'Chair is okay but assembly was difficult. Could be more comfortable.', '2024-01-22'),
  ('REV_003', 'CUST_003', 'Coffee Machine', 'Perfect coffee every morning! Easy to use and clean. Highly recommend.', '2024-01-25'),
  ('REV_004', 'CUST_004', 'Gaming Mouse', 'Great precision for gaming but the scroll wheel is a bit loose.', '2024-01-28'),
  ('REV_005', 'CUST_001', 'Standing Desk', 'Terrible quality. Desk wobbles and paint is chipping. Very disappointed!', '2024-01-30'),
  ('REV_006', 'CUST_002', 'Laptop Stand', 'Good value for money. Works as expected.', '2024-02-01'),
  ('REV_007', 'CUST_003', 'Coffee Mug', 'Beautiful design and keeps coffee hot. Absolutely love it!', '2024-02-03');
```

### Simple AI Query Test

First, test if ai_query works in your workspace:

```sql
-- Test ai_query function
SELECT ai_query(
    'databricks-meta-llama-3-3-70b-instruct',
    'Analyze this text and return only: positive, negative, or neutral. Text: I love this product!'
) as sentiment_test;
```


### Customer Reviews with AI Analysis

Create your customer reviews table and use ai_query for intelligent analysis:

```sql
-- Customer Reviews with AI Query Analysis
CREATE OR REPLACE VIEW sales_analytics.views.ai_customer_insights AS
SELECT 
    r.review_id,
    r.customer_id,
    c.customer_name,
    c.city,
    r.product_name,
    p.category,
    p.brand,
    r.review_text,
    r.review_date,
    
    -- AI-powered sentiment analysis
    ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'Analyze this customer review and return only one word: positive, negative, or neutral. Review: ' || r.review_text
    ) as ai_sentiment,
    
    -- AI-powered summary generation
    ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'Summarize this customer review in 10 words or less: ' || r.review_text
    ) as ai_summary,
    
    -- AI-powered urgency assessment
    ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        'Rate the urgency of addressing this customer feedback on a scale 1-5 (1=low, 5=urgent). Return only the number: ' || r.review_text
    ) as urgency_score

FROM sales_analytics.sales_data.customer_reviews r
LEFT JOIN sales_analytics.sales_data.customers c ON r.customer_id = c.customer_id
LEFT JOIN sales_analytics.sales_data.products p ON r.product_name = p.product_name;
```


### Dashboard Queries with AI Insights

**AI Sentiment Distribution:**

```sql
-- Customer sentiment analysis dashboard
SELECT 
    ai_sentiment,
    COUNT(*) as review_count,
    AVG(CAST(urgency_score AS INT)) as avg_urgency
FROM sales_analytics.views.ai_customer_insights
WHERE ai_sentiment IN ('Positive', 'Negative', 'Neutral')
GROUP BY ai_sentiment;
```
