WITH ru_policy_ids(id) AS (
  SELECT unnest(ARRAY[
    '00d3ee16-b7e5-4118-a155-d39fc701fa4b', '0f18c4dc-eacd-4bee-b6b6-f08eb5c963d2', '0fa468c0-eb07-4417-be08-f4054ae9fbc7',
    '11aac5b6-3554-45a3-b2b3-baeb0cda3805', '12a5b96c-55f4-4610-9ff8-cc6494ea0e2b', '14d49057-99f0-412b-8536-c8aba2955ea8',
    '18d2a3a8-8f66-4eb6-a66d-77b194f97fde', '21a13e2e-686c-44f7-baa6-6131af8136ce', '2b509a1e-bc9c-44d1-965d-bc0fa086b25b',
    '3b7951b8-c7f3-441f-a577-ef133e09d80b', '3d653739-214a-4bc7-af8b-85d704a0800a', '3deda84d-a1d3-40bc-99e0-f8d58283ff3a',
    '3fef7e95-76cd-4c6e-85ad-df2a5a17815c', '4aae3d81-0545-44f3-8267-b16cfac2cf34', '4c3e28ea-2ea6-406d-b53f-808b3481690f',
    '5380b64e-f62e-43a2-ac52-30eba3647416', '55f7bc83-f154-41a1-bacf-9ddc48276c5d', '5775f965-30c1-4430-a255-2223a068743c',
    '60618b0d-8fa9-4262-9899-f7e6d7c01c97', '66159296-9225-42c6-a2a2-36f9682ebba9', '697666ce-dcab-4c2d-90d8-1984e2ada5cb',
    '6a5d1a8d-02f7-4230-a520-4c9f2883a295', '70cb0072-d679-45e0-8ab3-51867b110f45', '7120833e-9253-429a-8b54-fa3cba7e3ea7',
    '7454ea88-e96a-4773-b690-fb83bcea04da', '7479d41b-88ea-43b5-9ab3-6c7ae78efa7e', '76b5f888-23cf-44b7-90fc-49510ae6154b',
    '81178eb9-2d86-4ec9-b3ee-edcff42520c1', '888d83bc-80fc-430c-910f-a7b93b240822', '8b47f4ba-acd1-4b6d-b4cf-e561bbbd3fce',
    '8bfc3e5a-8750-45a8-babc-e9438a005716', '9179fd26-81fb-4a7a-8d8b-78a34d87115c', '98df9961-6eea-4827-869c-8aadc50e167b',
    '98fcbcbf-4887-47ff-80fd-2fe4083740d0', '997dbac6-2c17-44f4-9aa1-2051ea7f44f2', '9b49418f-c375-4c75-b936-7a1aca944946',
    '9fe2fcdd-3b45-45cb-8874-0f8a1a225837', 'a163d413-0208-4763-8ad0-b67fc4687a90', 'a4526679-0c28-47f2-a078-a9319fd85614',
    'a47ee93b-4130-47fa-b29a-47e1ee0df483', 'a50a53c9-e575-4bb3-8bca-78dbc5c8a921', 'a9232b61-a7d6-4ac0-908d-cd61c09c4e1a',
    'b1e1d790-a18f-401c-a60a-e20a83c4250a', 'b2df4783-62eb-4eeb-ab57-94429de522ec', 'b2ea9660-c2d3-47cf-b0a3-1728c1abb3df',
    'b380f1c2-f1f4-4e75-bf22-5432b72aabd8', 'b393d3c3-550e-491b-9979-8d553aa40f78', 'bdc15261-b594-4fb5-9878-61a4c21a491d',
    'c34fd03e-6f48-4c40-9b70-f344dfaa4912', 'cd4d0ec8-9ae7-4088-87a6-d9330242ee62', 'd1e21484-308d-4f5b-8f54-ec84cd50791c',
    'd87e02e6-48d3-4314-88d5-a27df3e2b725', 'dc9b806b-3a67-491f-9315-bf73820d850d', 'e02092ac-e1f3-4d19-abfb-2ab4e9591c91',
    'ea154fe5-e069-4f50-86e0-e63836d70f2e', 'ef7854e5-bf38-47a2-b0d2-3dbfeba0aa37', 'f242a144-7d20-4dd6-86e8-a58b82e7626f',
    'f7f34adc-0f88-4bdf-a8e5-12c3945fdb29', 'f9296b56-6fd3-4c90-8165-fcff0833c925'
  ]::uuid[])
),
visits AS (
  SELECT
    vi.visit_id,
    vi.type,
    vi.date_of_visit
  FROM core.v_invoice vi
  JOIN core.visit        v  ON v.id  = vi.visit_id
  JOIN core.case_info    ci ON ci.id = v.case_id
  JOIN ru_policy_ids     rp ON rp.id = ci.policy_name_id
  WHERE vi.type <> 'ADMIN COST'::core.visit_type
),
spine AS (
  SELECT d::date AS obs_date
  FROM generate_series(DATE '2023-07-01', DATE '2026-07-31', INTERVAL '1 day') d
)
SELECT
  s.obs_date,
  COUNT(v.visit_id) AS total_cases
FROM spine s
LEFT JOIN visits v ON v.date_of_visit = s.obs_date
GROUP BY s.obs_date
ORDER BY s.obs_date;